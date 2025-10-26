"""
Smoke test helper that exercises the Python gRPC server via grpc stubs.

Usage:
    uv run python scripts/test_grpc_calls.py --file /absolute/path/to/video.mp4
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import grpc

from protos import video_analyzer_pb2 as pb
from protos import video_analyzer_pb2_grpc as pb_grpc


def register_local_video(
    client: pb_grpc.VideoAnalyzerServiceStub,
    file_path: pathlib.Path,
    display_name: Optional[str],
    reference_only: bool,
) -> pb.RegisterVideoResponse:
    request = pb.RegisterVideoRequest(
        file_path=str(file_path),
        display_name=display_name or "",
        reference_only=reference_only,
    )
    return client.RegisterLocalVideo(request)


def send_chat_message(
    client: pb_grpc.VideoAnalyzerServiceStub,
    message: str,
    file_id: str,
) -> None:
    chat_request = pb.ChatRequest(message=message, file_id=file_id)
    for chunk in client.SendChatMessage(chat_request):
        print("ChatResponse:", chunk)


def get_chat_history(
    client: pb_grpc.VideoAnalyzerServiceStub,
    limit: int,
) -> None:
    history = client.GetChatHistory(pb.HistoryRequest(limit=limit))
    print("ChatHistoryResponse:", history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test all gRPC endpoints end-to-end.")
    parser.add_argument("--addr", default="localhost:50051", help="gRPC server address.")
    parser.add_argument("--file", required=True, help="Absolute path to a local video.")
    parser.add_argument(
        "--display-name",
        default="Test Clip",
        help="Optional name to register alongside the file.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Do not copy the file into managed storage.",
    )
    parser.add_argument(
        "--chat-message",
        default="Give me a quick summary.",
        help="Chat prompt to send after registration.",
    )
    parser.add_argument("--history-limit", type=int, default=5)
    args = parser.parse_args()

    video_path = pathlib.Path(args.file).expanduser().resolve()
    if not video_path.is_file():
        print(f"[error] File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    channel = grpc.insecure_channel(args.addr)
    client = pb_grpc.VideoAnalyzerServiceStub(channel)

    print(f"Registering local file: {video_path}")
    register_resp = register_local_video(
        client=client,
        file_path=video_path,
        display_name=args.display_name,
        reference_only=args.reference_only,
    )
    print("RegisterLocalVideo:", register_resp)

    if not register_resp.file_id:
        print("[error] RegisterLocalVideo did not return a file_id.", file=sys.stderr)
        sys.exit(1)

    print(f"Sending chat message: {args.chat_message}")
    send_chat_message(client, args.chat_message, register_resp.file_id)

    print(f"Fetching chat history (limit={args.history_limit})")
    get_chat_history(client, args.history_limit)


if __name__ == "__main__":
    main()
