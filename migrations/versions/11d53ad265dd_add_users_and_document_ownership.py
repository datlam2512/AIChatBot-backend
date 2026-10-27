"""add users and document ownership

Revision ID: 11d53ad265dd
Revises: 
Create Date: 2026-10-27 22:53:51.610514

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '11d53ad265dd'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    # 1. Tạo bảng users
    op.create_table(
        "users",
        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column(
            "email",
            sa.String(255),
            nullable=False,
        ),
        sa.Column(
            "password_hash",
            sa.String(255),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "email",
            name="uq_users_email",
        ),
    )

    # 2. Thêm user_id vào conversations
    op.add_column(
        "conversations",
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=True,
        ),
    )

    op.create_foreign_key(
        "fk_conversations_user_id",
        "conversations",
        "users",
        ["user_id"],
        ["id"],
    )

    op.create_index(
        "ix_conversations_user_id",
        "conversations",
        ["user_id"],
    )

    # 3. Thêm user_id vào documents
    op.add_column(
        "documents",
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=True,
        ),
    )

    op.create_foreign_key(
        "fk_documents_user_id",
        "documents",
        "users",
        ["user_id"],
        ["id"],
    )

    op.create_index(
        "ix_documents_user_id",
        "documents",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documents_user_id",
        table_name="documents",
    )

    op.drop_constraint(
        "fk_documents_user_id",
        "documents",
        type_="foreignkey",
    )

    op.drop_column("documents", "user_id")

    op.drop_index(
        "ix_conversations_user_id",
        table_name="conversations",
    )

    op.drop_constraint(
        "fk_conversations_user_id",
        "conversations",
        type_="foreignkey",
    )

    op.drop_column("conversations", "user_id")

    op.drop_table("users")
