"""adoption health and fur fields

Revision ID: b7e2c41f9a10
Revises: 4d32595429e5
Create Date: 2026-10-18 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e2c41f9a10'
down_revision: Union[str, Sequence[str], None] = '4d32595429e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('pets', sa.Column('vaccinated', sa.Boolean(), nullable=True))
    op.add_column('pets', sa.Column('dewormed', sa.Boolean(), nullable=True))
    op.add_column('pets', sa.Column('sterilized', sa.Boolean(), nullable=True))
    op.add_column('pets', sa.Column('fur_length', sa.String(length=10), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('pets', 'fur_length')
    op.drop_column('pets', 'sterilized')
    op.drop_column('pets', 'dewormed')
    op.drop_column('pets', 'vaccinated')
