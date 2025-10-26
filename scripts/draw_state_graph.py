from orchestrator import MultiStageOrchestrator
from pathlib import Path
import platform

graph = MultiStageOrchestrator().workflow.get_graph(xray=True)
graph_image = graph.draw_mermaid_png()

def get_downloads_dir() -> Path:
    home = Path.home()
    system = platform.system()
    if system == "Windows":
        return home / "Downloads"
    if system == "Darwin":
        return home / "Downloads"
    return home / "Downloads"

downloads_dir = get_downloads_dir()
downloads_dir.mkdir(parents=True, exist_ok=True)
output_path = downloads_dir / "workflow_graph.png"

with open(output_path, "wb") as f:
    f.write(graph_image)

print(f"Saved workflow_graph.png to {output_path}")
