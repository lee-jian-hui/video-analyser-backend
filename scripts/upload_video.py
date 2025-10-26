#!/usr/bin/env python3
"""
CLI helper for streaming uploads to the VideoAnalyzer gRPC server.

Usage:
    uv run python scripts/upload_video.py path/to/video.mp4 \
        --address localhost:50051 \
        --chunk-size 1048576
"""

import argparse
import os
from typing import Iterator

import grpc

from protos import video_analyzer_pb2, video_analyzer_pb2_grpc


def iter_chunks(path: str, chunk_size: int) -> Iterator[video_analyzer_pb2.VideoChunk]:
    """Yield VideoChunk messages for the given file."""
    filename = os.path.basename(path)
    chunk_index = 0

    with open(path, "rb") as source:
        while True:
            data = source.read(chunk_size)
            if not data:
                break

            yield video_analyzer_pb2.VideoChunk(
                data=data,
                filename=filename,
                chunk_index=chunk_index,
            )
            chunk_index += 1


def upload_video(address: str, path: str, chunk_size: int) -> video_analyzer_pb2.UploadResponse:
    """Stream the file to the UploadVideo RPC and return the response."""
    channel = grpc.insecure_channel(address)
    stub = video_analyzer_pb2_grpc.VideoAnalyzerServiceStub(channel)
    return stub.UploadVideo(iter_chunks(path, chunk_size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a video file to the gRPC backend.")
    parser.add_argument("path", help="Path to the video file to upload.")
    parser.add_argument(
        "--address",
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Chunk size in bytes for each VideoChunk message (default: 1 MiB).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(f"File not found: {args.path}")

    response = upload_video(args.address, args.path, args.chunk_size)
    print("✅ Upload complete")
    print(f"   File ID : {response.file_id}")
    print(f"   Success : {response.success}")
    print(f"   Message : {response.message}")


if __name__ == "__main__":
    main()
