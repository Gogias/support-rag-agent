import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # корень проекта, чтобы работали импорты src.agent...

from src.agent.graph import graph as agent_graph  # noqa: E402

OUT = Path(__file__).parent / "graph"

graph = agent_graph.get_graph()

mermaid = graph.draw_mermaid()
OUT.with_suffix(".mmd").write_text(mermaid, encoding="utf-8")
print(mermaid)

try:
    graph.draw_mermaid_png(output_file_path=str(OUT.with_suffix(".png")))
    print(f"PNG: {OUT.with_suffix('.png')}")
except Exception as e:
    # PNG рендерится через mermaid.ink; без доступа к нему вставьте .mmd на https://mermaid.live
    print(f"PNG не создан ({e}). Mermaid-код сохранён в {OUT.with_suffix('.mmd')}")