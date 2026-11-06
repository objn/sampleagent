"""Run queued RAG ingestion jobs outside the HTTP request process."""

import argparse
import logging
import time

import psycopg

from Models.postgresql import claim_upload, complete_upload, fail_upload
from RAG.knowledge import ingest_file
from RAG.uploads import read_queued_upload


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_one() -> bool:
    job = claim_upload()
    if job is None:
        return False
    upload_id = job["id"]
    try:
        data = read_queued_upload(job["stored_name"], job["sha256"])
        result = ingest_file(job["original_name"], data, job["category"])
        complete_upload(upload_id, result)
        logger.info("RAG upload %s completed", upload_id)
    except ValueError as exc:
        fail_upload(upload_id, str(exc))
        logger.warning("RAG upload %s failed: %s", upload_id, exc)
    except Exception:
        fail_upload(upload_id, "Processing failed; see worker logs")
        logger.exception("RAG upload %s failed", upload_id)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Process pending RAG uploads")
    parser.add_argument("--once", action="store_true", help="Process at most one upload")
    args = parser.parse_args()
    while True:
        try:
            processed = process_one()
        except (psycopg.Error, OSError):
            logger.exception("RAG worker cannot reach storage")
            processed = False
        if args.once:
            return
        if not processed:
            time.sleep(2)


if __name__ == "__main__":
    main()
