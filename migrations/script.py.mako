"""${message}

Revision ID: ${up_revision}
Revises:     ${down_revision | comma,n}
Created:     ${create_date}

Description
-----------
<Describe the purpose of this migration and any operational notes.>

Rollback Notes
--------------
<List any data loss risks or manual steps needed before downgrading.>
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
${imports if imports else ""}
from alembic import op

# ---------------------------------------------------------------------------
# Revision identifiers — used by Alembic.
# ---------------------------------------------------------------------------

revision: str = ${repr(up_revision)}
down_revision: Union[str, Sequence[str], None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


# ---------------------------------------------------------------------------
# Schema changes
# ---------------------------------------------------------------------------

def upgrade() -> None:
    """Apply forward migration."""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Revert this migration.

    WARNING: Dropping tables is destructive.  Back up data before running
    ``alembic downgrade`` in a production environment.
    """
    ${downgrades if downgrades else "pass"}
