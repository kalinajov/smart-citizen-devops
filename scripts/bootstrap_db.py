"""
Create the schema and insert the lookup rows the API cannot start empty without.

The project ships no migration tooling: scripts/seed.py is commented out in its
entirety and scripts/migrate.sql only carries one ALTER TABLE patch, so it
assumes a schema that already exists. That is fine against the hand-built
Supabase instance, but a fresh PostgreSQL — the Compose volume, or the
StatefulSet in k8s/ — comes up empty and every endpoint 500s with
`relation "categories" does not exist`.

This script closes that gap. It is idempotent: safe to run on every deploy, and
it is what the Kubernetes bootstrap Job and the Compose quickstart both call.

Usage:
    python scripts/bootstrap_db.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from uuid import UUID

# Allow running as a plain script from the repo root or from /app in the image.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.base import Base  # noqa: F401 — imports every model onto Base.metadata
from app.db.session import SessionLocal, engine
from app.models.category import Category
from app.models.status import Status
from app.models.user import User, UserRole

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("bootstrap")


# Referenced by app/services/ai_service.py, which falls back to the category
# named by AI_DEFAULT_CATEGORY_NAME ("Other") when classification is off.
CATEGORIES = [
    (1, "Infrastructure"),
    (2, "Environment"),
    (3, "Safety"),
    (4, "Other"),
]

# app/utils/report_statuses.py drives the workflow off these names.
STATUSES = [
    (1, "Submitted"),
    (2, "In Progress"),
    (3, "Resolved"),
    (4, "Rejected"),
    (5, "Closed"),
    (6, "Pending"),
]

# The first entry is DEV_USER from app/utils/dependencies.py. With
# DEV_SKIP_AUTH=true every request is attributed to that id, so the row has to
# exist or each insert fails its foreign key.
USERS = [
    (UUID("12345678-1234-1234-1234-123456789012"), "dev@example.com", UserRole.admin),
    (UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"), "officer@example.com", UserRole.officer),
    (UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"), "citizen@example.com", UserRole.citizen),
]


def create_schema() -> None:
    """Create any missing tables. Existing tables are left untouched."""
    logger.info("Creating tables (if missing)...")
    Base.metadata.create_all(engine)
    logger.info("Schema ready: %d tables declared.", len(Base.metadata.tables))


def seed_lookups() -> None:
    """Insert reference data, skipping rows that are already present."""
    with SessionLocal() as db:
        added = 0

        for cat_id, name in CATEGORIES:
            if db.get(Category, cat_id) is None:
                db.add(Category(id=cat_id, name=name))
                added += 1

        for status_id, name in STATUSES:
            if db.get(Status, status_id) is None:
                db.add(Status(id=status_id, name=name))
                added += 1

        for user_id, email, role in USERS:
            # Match on id or email: both carry a unique constraint, so either
            # collision would abort the transaction.
            exists = db.execute(
                select(User).where((User.id == user_id) | (User.email == email))
            ).scalar_one_or_none()
            if exists is None:
                db.add(User(id=user_id, email=email, role=role))
                added += 1

        db.commit()
        logger.info("Seed complete: %d new row(s).", added)


def main() -> int:
    try:
        create_schema()
        seed_lookups()
    except SQLAlchemyError:
        logger.exception("Bootstrap failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
