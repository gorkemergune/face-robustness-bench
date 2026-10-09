"""Draw the illustrative (imaginary) face cards used in the README.

No real person or photo is involved: every card is the same cartoon face
drawn as SVG, with one condition from the benchmark applied.

    python scripts/make_readme_art.py   # writes docs/assets/*.svg
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("docs/assets")
SKIN, HAIR, INK = "#F1C7A3", "#4A3426", "#3B2A20"


def face(
    skin: str = SKIN,
    hair: str = HAIR,
    iris: str = "#5B4636",
    brows: str = "M70 86 Q82 79 94 85 M106 85 Q118 79 130 86",
    hair_path: str = "M46 104 Q44 44 100 40 Q156 44 154 104 Q146 70 120 62 Q96 74 64 70 Q52 82 46 104 Z",
    mouth: str = '<path d="M84 140 Q100 152 116 140" stroke="#9A4B3C" stroke-width="3.5" fill="none" stroke-linecap="round"/>',
    extra_below: str = "",
    extra_above: str = "",
    dx: float = 0.0,
    scale: float = 1.0,
) -> str:
    """One cartoon head; ``extra_below`` goes under the hair, ``extra_above`` on top."""
    return f"""<g transform="translate({dx} 0) translate(100 120) scale({scale}) translate(-100 -120)">
  <rect x="84" y="158" width="32" height="34" rx="6" fill="{skin}"/>
  <path d="M40 240 Q46 186 100 182 Q154 186 160 240 Z" fill="#6C8EBF"/>
  <ellipse cx="47" cy="112" rx="9" ry="14" fill="{skin}"/>
  <ellipse cx="153" cy="112" rx="9" ry="14" fill="{skin}"/>
  <ellipse cx="100" cy="108" rx="54" ry="64" fill="{skin}"/>
  {extra_below}
  <path d="{hair_path}" fill="{hair}"/>
  <path d="{brows}" stroke="{INK}" stroke-width="4" fill="none" stroke-linecap="round"/>
  <ellipse cx="82" cy="102" rx="10" ry="7" fill="#fff"/>
  <ellipse cx="118" cy="102" rx="10" ry="7" fill="#fff"/>
  <circle cx="82" cy="102" r="5" fill="{iris}"/>
  <circle cx="118" cy="102" r="5" fill="{iris}"/>
  <circle cx="83.5" cy="100.5" r="1.6" fill="#fff"/>
  <circle cx="119.5" cy="100.5" r="1.6" fill="#fff"/>
  <path d="M100 108 Q96 122 100 126 Q104 127 106 124" stroke="#C99474" stroke-width="3" fill="none" stroke-linecap="round"/>
  {mouth}
  {extra_above}
</g>"""


def card(name: str, title: str, subtitle: str, body: str, bg: str = "#EEF2F8", defs: str = "") -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 270" width="200" height="270" role="img" aria-label="{title}: {subtitle} (illustration)">
<defs>
  <clipPath id="frame-{name}"><rect x="8" y="8" width="184" height="200" rx="14"/></clipPath>
  {defs}
</defs>
<rect width="200" height="270" rx="18" fill="#FFFFFF"/>
<rect x="8" y="8" width="184" height="200" rx="14" fill="{bg}"/>
<g clip-path="url(#frame-{name})">{body}</g>
<text x="100" y="236" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="17" font-weight="700" fill="#1F2937">{title}</text>
<text x="100" y="256" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#6B7280">{subtitle}</text>
</svg>
"""
    (OUT / f"{name}.svg").write_text(svg)


def sparkle(x: float, y: float, r: float, color: str = "#FFFFFF") -> str:
    return (f'<path d="M{x} {y - r} L{x + r * .25} {y - r * .25} L{x + r} {y} L{x + r * .25} {y + r * .25} '
            f'L{x} {y + r} L{x - r * .25} {y + r * .25} L{x - r} {y} L{x - r * .25} {y - r * .25} Z" fill="{color}"/>')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    card("clean", "Clean", "enrollment photo", face())

    card(
        "filter", "Filter", "beauty / AR filter",
        face(extra_above=(
            '<path d="M58 58 L44 18 L80 44 Z M142 58 L156 18 L120 44 Z" fill="#E7A9C6"/>'
            '<path d="M60 52 L52 30 L72 46 Z M140 52 L148 30 L128 46 Z" fill="#F7D3E3"/>'
            '<ellipse cx="72" cy="124" rx="11" ry="6" fill="#F28FB5" opacity=".55"/>'
            '<ellipse cx="128" cy="124" rx="11" ry="6" fill="#F28FB5" opacity=".55"/>'
            + sparkle(36, 150, 9) + sparkle(166, 70, 7) + sparkle(160, 160, 5)))
        + '<rect width="200" height="220" fill="#F5A3C7" opacity=".22"/>',
        bg="#FBE7F1",
    )

    card(
        "makeup", "Makeup", "heavy makeup",
        face(
            brows="M68 86 Q82 76 95 84 M105 84 Q118 76 132 86",
            mouth='<path d="M84 140 Q92 134 100 138 Q108 134 116 140 Q100 154 84 140 Z" fill="#C2185B"/>',
            extra_below=(
                '<ellipse cx="82" cy="95" rx="15" ry="8" fill="#8E5CC2" opacity=".55"/>'
                '<ellipse cx="118" cy="95" rx="15" ry="8" fill="#8E5CC2" opacity=".55"/>'
                '<ellipse cx="70" cy="126" rx="12" ry="7" fill="#E57373" opacity=".45"/>'
                '<ellipse cx="130" cy="126" rx="12" ry="7" fill="#E57373" opacity=".45"/>'),
            extra_above='<path d="M71 100 L64 96 M129 100 L136 96" stroke="#111" stroke-width="2.5" stroke-linecap="round"/>',
        ),
        bg="#F3E8FB",
    )

    stripes = "".join(
        f'<rect x="{dx}" y="{y}" width="200" height="{h}" fill="{c}" opacity=".45"/>'
        for dx, y, h, c in [(-8, 70, 8, "#00E5FF"), (10, 96, 5, "#FF2BD6"), (-14, 132, 7, "#00E5FF"), (6, 150, 4, "#FF2BD6")]
    )
    card("effect", "Effect", "glitch / distortion",
         f'<g opacity=".5" transform="translate(-6 0)">{face()}</g><g transform="translate(5 0)">{face()}</g>{stripes}',
         bg="#E6F7FB")

    card("lens", "Lens", "colored contact lenses",
         face(iris="#1E88E5", extra_above=(
             '<circle cx="82" cy="102" r="6.5" fill="none" stroke="#64B5F6" stroke-width="1.5"/>'
             '<circle cx="118" cy="102" r="6.5" fill="none" stroke="#64B5F6" stroke-width="1.5"/>')),
         bg="#E7F1FC")

    card("injury", "Injury", "bandage / bruise",
         face(extra_above=(
             '<ellipse cx="124" cy="104" rx="16" ry="12" fill="#8E6BA8" opacity=".35"/>'
             '<g transform="rotate(-28 76 128)"><rect x="56" y="120" width="40" height="15" rx="7" fill="#F2D2A9" stroke="#C9A27A"/>'
             '<rect x="70" y="120" width="12" height="15" fill="#E8C292"/></g>'
             '<path d="M108 66 l8 10 M114 64 l6 9" stroke="#B5534B" stroke-width="2" stroke-linecap="round"/>')),
         bg="#FBF0E6")

    card("bad", "Bad photo", "crowd / half face / blur",
         f'<g filter="url(#blur-bad)">{face(dx=-40)}</g>'
         + face(dx=48, scale=1.15, skin="#D9A57E", hair="#1F1A17",
                hair_path="M46 100 Q48 40 100 38 Q152 40 154 100 Q150 58 100 56 Q50 58 46 100 Z"),
         bg="#ECEFF3",
         defs='<filter id="blur-bad"><feGaussianBlur stdDeviation="2.2"/></filter>')

    card("twin", "AI twin", "look-alike (not the person)",
         face(
             hair="#5A3E2B",
             brows="M68 88 Q82 82 95 86 M105 86 Q118 82 132 88",
             hair_path="M46 104 Q40 50 96 38 Q160 40 154 104 Q150 64 100 66 Q62 60 46 104 Z",
             mouth='<path d="M86 141 Q100 148 114 141" stroke="#9A4B3C" stroke-width="3.5" fill="none" stroke-linecap="round"/>',
             extra_above=(
                 '<circle cx="128" cy="130" r="2.6" fill="#6D4C41"/>'
                 '<rect x="138" y="18" width="44" height="24" rx="12" fill="#7C3AED"/>'
                 '<text x="160" y="35" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
                 'font-size="13" font-weight="700" fill="#fff">AI</text>'),
         ),
         bg="#EFEAFB")
    print(f"wrote {len(list(OUT.glob('*.svg')))} cards to {OUT}")


if __name__ == "__main__":
    main()
