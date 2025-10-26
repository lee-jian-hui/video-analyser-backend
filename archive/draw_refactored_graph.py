"""Draw the refactored orchestrator state graph"""
from orchestrator_refactored import MultiStageOrchestrator
from pathlib import Path
import platform

# Initialize orchestrator
orchestrator = MultiStageOrchestrator()

# Get graph visualization
graph = orchestrator.workflow.get_graph()

# Draw as PNG
png_data = graph.draw_mermaid_png()

# Determine save location based on platform
if platform.system() == "Windows":
    # WSL: Save to Windows Downloads folder
    home = Path.home()
    downloads = home / "Downloads"
    if not downloads.exists():
        # Fallback to current directory
        downloads = Path(".")
    output_path = downloads / "refactored_workflow_graph.png"
else:
    # Linux/Mac: Save to Downloads or current directory
    downloads = Path.home() / "Downloads"
    if not downloads.exists():
        downloads = Path(".")
    output_path = downloads / "refactored_workflow_graph.png"

# Save the PNG
with open(output_path, "wb") as f:
    f.write(png_data)

print(f"Saved refactored workflow_graph.png to {output_path}")
