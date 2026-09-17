from __future__ import annotations

import logging
from time import perf_counter
from uuid import UUID
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.attachment import Attachment
from app.models.category import Category
from app.models.comment import Comment
from app.models.history import History
from app.models.report import Report
from app.models.status import Status
from app.schemas.attachment import AttachmentRead
from app.schemas.report import (
    CommentCreate,
    CommentRead,
    PriorityUpdate,
    ReportCreate,
    ReportRead,
    ReportUpdate,
    StatusUpdate,
)
from app.schemas.user import CurrentUser, UserRole
from app.models.user import User
from app.services.ai_service import assign_priority, classify_text, generate_confirmation_message, generate_confirmation_mk
from app.utils.duplicate_detection import check_duplicate
from app.utils.report_statuses import ACTIVE_STATUS_NAMES, status_ids_for_names

logger = logging.getLogger(__name__)

DEFAULT_SUBMITTED_STATUS = "Активен"


def _normalize_category_key(value: str) -> str:
    return value.strip().casefold()


def _classifier_label_for_category(category_name: str) -> str:
    if category_name.strip().lower() == "safety":
        return "Safety (accidents and hazards)"
    return category_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_404(db: Session, report_id: UUID) -> Report:
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


def _to_report_read(report: Report, db: Session | None = None) -> ReportRead:
    """
    Defensive conversion to ReportRead. Accepts real ORM objects or MagicMocks used in unit tests.
    """
    def _safe(val, cast=None):
        if val is None:
            return None
        try:
            return cast(val) if cast is not None else (str(val) if not isinstance(val, str) else val)
        except Exception:
            return None

    user_email = None
    try:
        user_email = getattr(getattr(report, "user", None), "email", None)
    except Exception:
        user_email = None

    if user_email is None and db is not None:
        try:
            u = db.get(User, getattr(report, "user_id", None))
            user_email = getattr(u, "email", None) if u is not None else None
        except Exception:
            user_email = None

    return ReportRead(
        id=getattr(report, "id", None),
        description=_safe(getattr(report, "description", None)),
        category_id=_safe(getattr(report, "category_id", None), int),
        status_id=_safe(getattr(report, "status_id", None), int),
        user_id=getattr(report, "user_id", None),
        user_email=user_email,
        latitude=getattr(report, "latitude", None),
        longitude=getattr(report, "longitude", None),
        priority=getattr(report, "priority", None),
        possible_duplicate_of=getattr(report, "possible_duplicate_of", None),
        ai_confirmation_text=getattr(report, "ai_confirmation_text", None),
        created_at=getattr(report, "created_at", datetime.now(timezone.utc)),
        updated_at=getattr(report, "updated_at", datetime.now(timezone.utc)),
    )

def create_report(db: Session, *, report_in: ReportCreate, current_user: CurrentUser) -> ReportRead:
    """Persist a new report and return the API view.

    AI classification + duplicate detection run asynchronously in
    `run_report_ai_pipeline`, scheduled by the route as a BackgroundTask.
    """
    report = Report(
        description=report_in.description,
        latitude=report_in.latitude,
        longitude=report_in.longitude,
        category_id=report_in.category_id,
        user_id=current_user.id,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return _to_report_read(report, db=db)

def run_report_ai_pipeline(report_id: UUID) -> None:
    """
    Background pipeline: classify category, assign priority, generate confirmation texts.

    Behavior is intentionally conservative and must not raise exceptions to avoid breaking report
    creation. Uses `SessionLocal()` and `get_settings()` — tests monkeypatch these.
    """
    total_start = perf_counter()
    db: Session = SessionLocal()
    try:
        settings = get_settings()
    except Exception:
        settings = None

    try:
        report = db.get(Report, report_id)
        if report is None:
            logger.warning("AI pipeline: report_id=%s not found — skipping.", report_id)
            return

        categories = db.scalars(select(Category)).all() if hasattr(db, "scalars") else []
        if not categories:
            logger.warning("AI pipeline: no categories found — skipping for report_id=%s.", report_id)
            categories_sorted: list[Category] = []
        else:
            categories_sorted = sorted(categories, key=lambda c: c.name.casefold())

        category_name_key_to_id = {_normalize_category_key(c.name): c.id for c in categories_sorted}
        classifier_label_key_to_id = {
            _normalize_category_key(_classifier_label_for_category(c.name)): c.id for c in categories_sorted
        }
        candidate_labels = [_classifier_label_for_category(c.name) for c in categories_sorted]

        predicted_label: str | None = None
        category_changed = False

        # Only auto-assign when the report still has no category.
        if report.category_id is None and candidate_labels:
            try:
                predicted_label = classify_text(
                    report.description,
                    candidate_labels,
                    min_confidence=getattr(settings, "ai_min_confidence", 0.5) if settings is not None else 0.5,
                )
            except Exception:
                logger.warning("AI unavailable — skipping for report_id=%s.", report_id, exc_info=True)
                predicted_label = None

        if report.category_id is None and predicted_label:
            predicted_key = _normalize_category_key(predicted_label)
            new_category_id = (
                category_name_key_to_id.get(predicted_key)
                or classifier_label_key_to_id.get(predicted_key)
            )
            if new_category_id is not None:
                report.category_id = new_category_id
                category_changed = True
                logger.info(
                    "Auto-assigned category_id=%d ('%s') for report_id=%s",
                    new_category_id, predicted_label, report_id,
                )
            else:
                logger.warning("AI label not matched to DB category: %s", predicted_label)

        if report.category_id is None and getattr(settings, "ai_default_category_name", None):
            fallback_id = category_name_key_to_id.get(
                _normalize_category_key(getattr(settings, "ai_default_category_name"))
            )
            if fallback_id is not None:
                report.category_id = fallback_id
                category_changed = True
                logger.info(
                    "Applied fallback category_id=%d ('%s') for report_id=%s",
                    fallback_id, getattr(settings, "ai_default_category_name"), report_id,
                )
            else:
                logger.warning(
                    "Fallback category '%s' not found in DB — left NULL for report_id=%s.",
                    getattr(settings, "ai_default_category_name"), report_id,
                )

        if category_changed:
            db.commit()

        # Priority assignment: do not override an existing non-empty priority
        if not (getattr(report, "priority", None) or "").strip():
            try:
                recent_descriptions = db.scalars(
                    select(Report.description)
                    .where(Report.id != report.id)
                    .order_by(Report.created_at.desc())
                    .limit(20)
                ).all()
                report.priority = assign_priority(report.description, list(recent_descriptions))
                db.commit()
                logger.info("Auto-assigned priority='%s' for report_id=%s", report.priority, report_id)
            except Exception:
                db.rollback()
                logger.warning(
                    "Priority assignment failed for report_id=%s — continuing pipeline.",
                    report_id,
                    exc_info=True,
                )

        category_label_for_message: str | None = None
        if report.category_id is not None and categories_sorted:
            category_label_for_message = next(
                (c.name for c in categories_sorted if c.id == report.category_id), None
            )

        try:
            message = generate_confirmation_message(
                report.description,
                category_label=category_label_for_message,
                possible_duplicate_of=getattr(report, "possible_duplicate_of", None),
            )
        except Exception:
            message = None
            logger.warning("generate_confirmation_message failed for report_id=%s", report_id, exc_info=True)

        try:
            mk_text = generate_confirmation_mk(
                report.description,
                category_label=category_label_for_message,
                priority=getattr(report, "priority", None),
                possible_duplicate_of=getattr(report, "possible_duplicate_of", None),
            )
            report.ai_confirmation_text = mk_text
            db.commit()
            logger.info("ai_confirmation_text saved for report_id=%s", report_id)
        except Exception:
            logger.warning(
                "Unexpected error saving ai_confirmation_text for report_id=%s — skipping.",
                report_id,
                exc_info=True,
            )

        if message and getattr(settings, "ai_confirmation_comment_user_id", None) is not None:
            existing = db.scalars(
                select(Comment)
                .where(Comment.report_id == report.id)
                .where(Comment.user_id == getattr(settings, "ai_confirmation_comment_user_id"))
            ).first()
            if existing is None:
                db.add(Comment(
                    report_id=report.id,
                    user_id=getattr(settings, "ai_confirmation_comment_user_id"),
                    content=message,
                ))
                try:
                    db.commit()
                except Exception:
                    db.rollback()
                    logger.warning(
                        "AI confirmation persistence failed for report_id=%s.", report_id, exc_info=True
                    )

        elapsed_ms = (perf_counter() - total_start) * 1000
        logger.info("AI pipeline latency: %.0fms", elapsed_ms)
    except Exception:
        logger.warning("AI pipeline failed for report_id=%s — skipping.", report_id, exc_info=True)
    finally:
        try:
            db.close()
        except Exception:
            pass

def list_reports(db: Session, *, current_user: CurrentUser) -> list[ReportRead]:
    try:
        stmt = select(Report).options(joinedload(Report.user))
        if current_user.role == UserRole.citizen:
            stmt = stmt.where(Report.user_id == current_user.id)
        elif current_user.role == UserRole.officer:
            active_status_ids = status_ids_for_names(db, ACTIVE_STATUS_NAMES)
            if not active_status_ids:
                return []
            stmt = stmt.where(Report.status_id.in_(active_status_ids))
        # Admin sees all
        rows = db.scalars(stmt).all()
        return [_to_report_read(r, db=db) for r in rows]
    except Exception as exc:
        logger.warning(
            "list_reports failed for user_id=%s role=%s; returning empty list: %s",
            getattr(current_user, "id", None), exc,
        )
        return []


def _record_status_history(
    db: Session,
    report: Report,
    new_status_id: int | None,
    changed_by_user_id: UUID,
) -> None:
    db.add(History(
        report_id=report.id,
        old_status_id=report.status_id,
        status_id=new_status_id,
        changed_by_user_id=changed_by_user_id,
    ))


def get_report(db: Session, *, report_id: UUID, current_user: CurrentUser) -> ReportRead:
    report = _get_or_404(db, report_id)
    if current_user.role == UserRole.citizen and report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")
    if current_user.role == UserRole.officer:
        active_status_ids = status_ids_for_names(db, ACTIVE_STATUS_NAMES)
        if not active_status_ids or report.status_id not in active_status_ids:
            raise HTTPException(status_code=403, detail="Not allowed")
    return _to_report_read(report, db=db)


def update_report(
    db: Session, *, report_id: UUID, report_in: ReportUpdate, current_user: CurrentUser
) -> ReportRead:
    report = _get_or_404(db, report_id)
    if current_user.role == UserRole.citizen:
        if report.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not allowed")
        update_data = report_in.model_dump() if hasattr(report_in, "model_dump") else {}
        # Citizens cannot change status or category
        update_data.pop("status_id", None)
        update_data.pop("category_id", None)
    elif current_user.role == UserRole.officer:
        raise HTTPException(status_code=403, detail="Not allowed")
    else:
        # admin
        update_data = report_in.model_dump() if hasattr(report_in, "model_dump") else {}

    new_status_id = update_data.get("status_id", getattr(report, "status_id", None))
    if new_status_id != getattr(report, "status_id", None):
        _record_status_history(db, report, new_status_id, current_user.id)

    for field, value in update_data.items():
        setattr(report, field, value)

    db.commit()
    db.refresh(report)
    return _to_report_read(report)


def update_status(
    db: Session, *, report_id: UUID, status_in: StatusUpdate, current_user: CurrentUser
) -> ReportRead:
    report = _get_or_404(db, report_id)
    status_changed = status_in.status_id != getattr(report, "status_id", None)
    if status_changed:
        _record_status_history(db, report, status_in.status_id, current_user.id)
    report.status_id = status_in.status_id

    if status_changed:
        new_status = db.get(Status, status_in.status_id)
        new_status_name = getattr(new_status, "name", None)
        if isinstance(new_status_name, str):
            try:
                from app.services.notification_service import (
                    create_status_change_notification,
                    create_rating_invitation_notification,
                )
                create_status_change_notification(db, report=report, new_status_name=new_status_name)
                if new_status_name.strip().casefold() in {"решен", "решена", "решено", "решени", "closed"}:
                    create_rating_invitation_notification(db, report=report)
            except Exception:
                logger.warning("Notification handlers failed", exc_info=True)

    db.commit()
    db.refresh(report)
    return _to_report_read(report)


def update_priority(
    db: Session,
    *,
    report_id: UUID,
    priority_in: PriorityUpdate,
    current_user: CurrentUser,
) -> ReportRead:
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Not allowed")
    report = _get_or_404(db, report_id)
    report.priority = priority_in.priority
    db.commit()
    db.refresh(report)
    return _to_report_read(report)


def delete_report(db: Session, *, report_id: UUID, current_user: CurrentUser) -> None:
    report = _get_or_404(db, report_id)
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Not allowed")
    db.delete(report)
    db.commit()


def list_comments(
    db: Session, *, report_id: UUID, current_user: CurrentUser
) -> list[CommentRead]:
    report = _get_or_404(db, report_id)
    if current_user.role == UserRole.citizen and report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")
    comments = db.scalars(
        select(Comment)
        .where(Comment.report_id == report_id)
        .order_by(Comment.created_at.asc())
    ).all()
    return [CommentRead.model_validate(c) for c in comments]


def create_comment(
    db: Session, *, report_id: UUID, comment_in: CommentCreate, current_user: CurrentUser
) -> CommentRead:
    if current_user.role not in [UserRole.officer, UserRole.admin]:
        raise HTTPException(status_code=403, detail="Only officers and admins can comment")

    report = _get_or_404(db, report_id)

    comment = Comment(
        report_id=report.id,
        user_id=current_user.id,
        content=comment_in.content,
    )
    db.add(comment)

    from app.services.notification_service import create_comment_notification
    create_comment_notification(db, report=report, commenter_user_id=current_user.id)

    db.commit()
    db.refresh(comment)
    return CommentRead.model_validate(comment)


def list_attachments(
    db: Session, *, report_id: UUID, current_user: CurrentUser
) -> list[AttachmentRead]:
    report = _get_or_404(db, report_id)
    if current_user.role == UserRole.citizen and report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed")
    attachments = db.scalars(
        select(Attachment)
        .where(Attachment.report_id == report_id)
        .order_by(Attachment.created_at.asc())
    ).all()
    return [AttachmentRead.model_validate(a) for a in attachments]


def create_attachment(
    db: Session,
    *,
    report_id: UUID,
    file_url: str,
    original_filename: str,
    content_type: str,
    file_size_bytes: int,
    current_user: CurrentUser,
) -> AttachmentRead:
    if current_user.role not in [UserRole.officer, UserRole.admin]:
        raise HTTPException(status_code=403, detail="Only officers and admins can upload attachments")

    report = _get_or_404(db, report_id)

    attachment = Attachment(
        report_id=report.id,
        file_url=file_url,
        original_filename=original_filename,
        content_type=content_type,
        file_size_bytes=file_size_bytes,
    )
    db.add(attachment)
    db.commit()
    db.refresh(attachment)
    return AttachmentRead.model_validate(attachment)

def export_reports(
    db: Session,
    *,
    status: str | None = None,
    category: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[Report]:
    """Return reports filtered for export, with category/status eagerly loaded.

    Filters are applied conjunctively. An unknown status/category name yields
    no matches (rather than being silently dropped) so the caller never gets
    back a wider set than they asked for.
    """
    stmt = (
        select(Report)
        .options(joinedload(Report.category), joinedload(Report.status))
        .order_by(Report.created_at.desc())
    )

    if status is not None:
        status_row = db.scalars(select(Status).where(Status.name == status)).first()
        if status_row is None:
            return []
        stmt = stmt.where(Report.status_id == status_row.id)

    if category is not None:
        category_row = db.scalars(select(Category).where(Category.name == category)).first()
        if category_row is None:
            return []
        stmt = stmt.where(Report.category_id == category_row.id)

    if date_from is not None:
        stmt = stmt.where(Report.created_at >= date_from)

    if date_to is not None:
        stmt = stmt.where(Report.created_at <= date_to)

    return list(db.scalars(stmt).unique().all())