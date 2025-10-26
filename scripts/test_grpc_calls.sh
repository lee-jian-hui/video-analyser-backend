#!/usr/bin/env bash
# Simple regression helper that exercises the Python gRPC server via grpcurl.
# Requires grpcurl + python (for JSON parsing). Set VIDEO_FILE_PATH env var.

set -euo pipefail

PROTO_PATH="protos/video_analyzer.proto"
ADDR="${GRPC_ADDR:-localhost:50051}"
VIDEO_PATH="${VIDEO_FILE_PATH:-}"
DISPLAY_NAME="${DISPLAY_NAME:-Sample Clip}"
REFERENCE_ONLY="${REFERENCE_ONLY:-false}"
CHAT_MESSAGE="${CHAT_MESSAGE:-\"Give me a quick summary\"}"
CHAT_HISTORY_LIMIT="${CHAT_HISTORY_LIMIT:-5}"

if [[ -z "${VIDEO_PATH}" ]]; then
  echo "VIDEO_FILE_PATH env var is required (absolute path to a test video)." >&2
  exit 1
fi

echo "📡 Using gRPC server at ${ADDR}"
echo "🎬 Registering local file: ${VIDEO_PATH}"

REGISTER_RESPONSE=$(grpcurl -plaintext \
  -proto "${PROTO_PATH}" \
  -d "{\"filePath\":\"${VIDEO_PATH}\",\"displayName\":\"${DISPLAY_NAME}\",\"referenceOnly\":${REFERENCE_ONLY}}" \
  "${ADDR}" \
  video_analyzer.VideoAnalyzerService/RegisterLocalVideo)

echo "🔁 RegisterLocalVideo response:"
echo "${REGISTER_RESPONSE}"

FILE_ID=$(python - <<'PY' "${REGISTER_RESPONSE}"
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("fileId", ""))
PY
)

if [[ -z "${FILE_ID}" ]]; then
  echo "Failed to extract fileId from RegisterLocalVideo response." >&2
  exit 1
fi

echo "✅ Registered file_id: ${FILE_ID}"
echo "💬 Sending chat request: ${CHAT_MESSAGE}"

grpcurl -plaintext \
  -proto "${PROTO_PATH}" \
  -d "{\"message\":${CHAT_MESSAGE},\"fileId\":\"${FILE_ID}\"}" \
  "${ADDR}" \
  video_analyzer.VideoAnalyzerService/SendChatMessage

echo ""
echo "📜 Fetching chat history (limit=${CHAT_HISTORY_LIMIT})"

grpcurl -plaintext \
  -proto "${PROTO_PATH}" \
  -d "{\"limit\":${CHAT_HISTORY_LIMIT}}" \
  "${ADDR}" \
  video_analyzer.VideoAnalyzerService/GetChatHistory

echo ""
echo "✅ All sample grpcurl calls completed."
