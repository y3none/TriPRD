"""Minimal TriPRD model package.

The implementation in :mod:`models.triprd` is intentionally self-contained:
it does not download a backbone or depend on a particular third-party COD
repository.  This makes it suitable for a quick feasibility run and gives the
training scripts a stable interface before a larger backbone is introduced.
"""

from .triprd import (
    TriPRDStudent,
    TriPRDTeacher,
    TriPRDLoss,
    teacher_gate,
    count_parameters,
)

from .triprd_full import TriPRDStudentFull, TriPRDTeacherFull

__all__ = ["TriPRDTeacher", "TriPRDStudent", "TriPRDLoss", "TriPRDTeacherFull", "TriPRDStudentFull", "teacher_gate", "count_parameters"]
