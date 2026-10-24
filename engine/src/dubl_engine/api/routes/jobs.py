from __future__ import annotations

from fastapi import APIRouter, Depends

from ...models import Job
from ...store import NotFoundError
from ..deps import JobsDep, require_token
from ..schemas import ErrorResponse

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_token)],
                   responses={404: {"model": ErrorResponse}})


@router.get("")
def list_jobs(jobs: JobsDep) -> list[Job]:
    return jobs.list()


@router.get("/{job_id}")
def get_job(job_id: str, jobs: JobsDep) -> Job:
    j = jobs.get(job_id)
    if j is None:
        raise NotFoundError("Задача не найдена.")
    return j


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, jobs: JobsDep) -> Job:
    j = jobs.cancel(job_id)
    if j is None:
        raise NotFoundError("Задача не найдена.")
    return j
