"""Draw one row of a colour-coded formula on a Matplotlib figure.

Matplotlib mathtext cannot colour parts of one expression, so a formula is
written as segments, each with its own colour. ``draw_math_row`` measures every
segment, shrinks the font once if the row is wider than ``max_width``, and
places the segments centred with a fixed gap.

Use English and math symbols only inside the figure; explain the colours in the
slide cards or narration.

Example::

    from math_row import draw_math_row
    draw_math_row(fig, 0.6, [(r"$z\\ =\\ ($", "#222222"), (r"$x$", "#0072B2"),
                             (r"$\\ -\\ $", "#222222"), (r"$\\mu$", "#D55E00"),
                             (r"$)$", "#222222")])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def measure_segments(fig: Figure, segments: Sequence[tuple[str, str]], fontsize: float) -> list[float]:
    """Width of each segment as a fraction of the figure width."""

    renderer = fig.canvas.get_renderer()
    widths = []
    for text, _ in segments:
        # 临时放一个文本对象量宽度，量完即删
        artist = fig.text(0, 0, text, fontsize=fontsize)
        widths.append(artist.get_window_extent(renderer=renderer).width / fig.bbox.width)
        artist.remove()
    return widths


def draw_math_row(
    fig: Figure,
    y: float,
    segments: Sequence[tuple[str, str]],
    fontsize: float = 30,
    gap: float = 0.012,
    max_width: float = 0.94,
) -> float:
    """Draw ``(text, colour)`` segments centred at height ``y``; return the font size used.

    ``y``, ``gap`` and ``max_width`` are fractions of the figure size.
    """

    if not segments:
        raise ValueError("draw_math_row needs at least one segment.")
    widths = measure_segments(fig, segments, fontsize)
    total = sum(widths) + gap * (len(segments) - 1)
    if total > max_width:
        # 片段宽度随字号线性变化，按比例缩小一次即可
        fontsize *= (max_width - gap * (len(segments) - 1)) / sum(widths)
        widths = measure_segments(fig, segments, fontsize)
        total = sum(widths) + gap * (len(segments) - 1)
    x = 0.5 - total / 2
    for (text, colour), width in zip(segments, widths):
        fig.text(x, y, text, fontsize=fontsize, color=colour, va="center")
        x += width + gap
    return fontsize
