"""Typeset the existing inference-route guide; no experimental inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_ref(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.as_posix(), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--font-regular", type=Path)
    parser.add_argument("--font-bold", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("The figure output directory must be fresh.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    if (args.font_regular is None) != (args.font_bold is None):
        raise ValueError('Supply both font paths, or neither.')
    if args.font_regular is None:
        args.font_regular = Path(font_manager.findfont(font_manager.FontProperties(family='STIXGeneral', weight='normal')))
        args.font_bold = Path(font_manager.findfont(font_manager.FontProperties(family='STIXGeneral', weight='bold')))
    for font in (args.font_regular, args.font_bold):
        font_manager.fontManager.addfont(str(font))
    regular = font_manager.FontProperties(fname=str(args.font_regular))
    bold = font_manager.FontProperties(fname=str(args.font_bold))
    regular.set_math_fontfamily("stix")
    bold.set_math_fontfamily("stix")
    plt.rcParams.update({
        "font.family": regular.get_name(), "font.size": 9.4,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.usetex": False, "savefig.facecolor": "white",
    })

    # Coordinates are points; export at the intended single-column width.
    fig = plt.figure(figsize=(3.45, 5.04))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 248.4)
    ax.set_ylim(-3, 360)
    ax.set_aspect("equal")
    ax.axis("off")
    ink = "#223044"
    connector = "#617082"
    content = []

    def text(x, y, value, size=9.4, strong=False, color=ink, linespacing=1.18):
        artist = ax.text(x, y, value, ha="center", va="center", rotation=0,
                         fontproperties=bold if strong else regular,
                         fontsize=size, color=color, linespacing=linespacing,
                         zorder=4)
        content.append(artist)
        return artist

    def box(x, y, width, height, fill, edge):
        patch = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0,rounding_size=4.5",
                              linewidth=0.8, edgecolor=edge, facecolor=fill,
                              zorder=2)
        ax.add_patch(patch)
        return (x, y, width, height)

    def line(xs, ys):
        ax.plot(xs, ys, color=connector, linewidth=0.85,
                solid_capstyle="round", solid_joinstyle="round", zorder=1)

    def down(x, y0, y1):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
                                    arrowstyle="-|>", mutation_scale=7,
                                    linewidth=0.85, color=connector,
                                    shrinkA=0, shrinkB=0, zorder=3))

    # Single top-level contract, then a symmetric orthogonal branch.
    box(4, 310, 240, 46, "#F3F5F7", "#778597")
    text(124, 343, "Define the scoring workload", 10.3, True)
    text(124, 325.5, "Fix graph, features, weights and targets.\n"
         "Set quality, memory and time constraints.", 9.4)
    down(124, 310, 288)
    box(45, 259, 158, 28, "#E9F0F7", "#617B96")
    text(124, 273, "Does exact computation fit?", 10.2, True)

    line([124, 124], [259, 236])
    line([61, 187], [236, 236])
    down(61, 236, 217)
    down(187, 236, 217)
    # Labels share the same baseline and sit above their horizontal branches.
    yes = text(92.5, 244.5, "Yes", 10.0, True)
    no = text(155.5, 244.5, "No", 10.0, True)

    box(4, 131, 114, 85, "#EEF4FA", "#63809C")
    text(61, 201, "Exact inference", 10.2, True)
    text(61, 169, "Share computation across\nall requested targets.\n"
         "Compare quality, time\nand peak memory.", 9.3, linespacing=1.24)
    box(130, 131, 114, 85, "#FAF4EB", "#9A8160")
    text(187, 201, "Sampled inference", 10.2, True)
    text(187, 169, "Compare Uniform / LABOR\non the same workload.\n"
         "Consider layerwise or\nout-of-core routes.", 9.3, linespacing=1.24)

    # Both routes feed the same paid-cache comparison.
    line([61, 61, 187, 187], [131, 118, 118, 131])
    down(124, 118, 106)
    box(4, 47, 240, 58, "#EEF5F0", "#758E7B")
    text(124, 92, "Reuse unchanged outputs", 10.2, True)
    text(124, 78, "Pay the exact build and cache if", 9.4)
    text(124, 60, r"$q > \frac{S_f+P_f-S_s}{T_s-T_c},\quad T_s>T_c$", 11.0)

    down(124, 47, 36)
    box(4, 0, 240, 35, "#F4F4F3", "#8B8D8E")
    text(124, 25, "Before adding a correction", 10.0, True)
    text(124, 10.5, "Compare matched training and\n"
         "equal-view native-logit averaging.", 9.3, linespacing=1.10)

    # The caption supplies the explanation; no internal title or footer.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_bounds = ax.get_window_extent(renderer)
    for artist in content:
        bounds = artist.get_window_extent(renderer)
        if (bounds.x0 < axis_bounds.x0 or bounds.x1 > axis_bounds.x1 or
                bounds.y0 < axis_bounds.y0 or bounds.y1 > axis_bounds.y1):
            raise RuntimeError("A text item exceeds the figure canvas.")
    assert yes.get_position()[1] == no.get_position()[1]
    assert yes.get_rotation() == no.get_rotation() == 0

    output.mkdir(parents=True, exist_ok=False)
    pdf = output / "final_decision_guide.pdf"
    png = output / "final_decision_guide.png"
    fig.savefig(pdf, metadata={"Title": "Inference-route decision guide",
                              "Creator": "Matplotlib; fixed conceptual layout"})
    fig.savefig(png, dpi=240)
    plt.close(fig)
    provenance = {
        "schema": "vldb_decision_guide_presentation_v1",
        "source": file_ref(Path(__file__).resolve()),
        "scope": "Conceptual figure presentation only; no result data read or recomputed.",
        "fonts": {"regular": file_ref(args.font_regular),
                  "bold": file_ref(args.font_bold), "math": "Matplotlib STIX Roman"},
        "matplotlib_version": matplotlib.__version__,
        "layout": {"units": "points", "page_inches": [3.45, 5.04],
                   "orthogonal_connectors": True, "yes_no_y": 244.5,
                   "label_rotation_degrees": 0, "internal_title": False,
                   "footer": False},
        "artifacts": [file_ref(pdf), file_ref(png)],
    }
    (output / "figure-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "output": str(output),
                      "artifacts": [file_ref(pdf), file_ref(png)]}))


if __name__ == "__main__":
    main()
