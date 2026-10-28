"""draws one SVG per decision process of the logistics agent, describing each by its operators

the text on each figure is written here by hand from the operators' WHEN/THEN docstrings, so redraw the figures after
changing an operator:

    python design-docs/figures/draw.py

writes the SVGs next to this script, or into the directory given as the first argument; uses only the standard library
"""

import sys
import textwrap
from html import escape
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

INK, MUTED, FAINT, LINE = "#0f172a", "#475569", "#64748b", "#94a3b8"
BADGES = {
    "llm": ("#ede9fe", "#6d28d9"),
    "world": ("#dcfce7", "#15803d"),
    "sub": ("#dbeafe", "#1d4ed8"),
    "turn": ("#fef3c7", "#b45309"),
    "stack": ("#fce7f3", "#be185d"),
}
CHAR = 6.5  # width of a 12px sans character, generously
MONO_CHAR = 7.4  # width of a 12px monospace character


def wrap(text: str, width: float, char: float = CHAR) -> list[str]:
    return textwrap.wrap(text, max(8, int(width / char)))


class Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def top(self, f=0.5):
        return (self.x + self.w * f, self.y)

    def bottom(self, f=0.5):
        return (self.x + self.w * f, self.y + self.h)

    def left(self, f=0.5):
        return (self.x, self.y + self.h * f)

    def right(self, f=0.5):
        return (self.x + self.w, self.y + self.h * f)


class Fig:
    def __init__(self, width: int, title: str, subtitle: str):
        self.w = width
        self.parts: list[str] = []
        self.overlay: list[str] = []
        self.bottom = 0
        self.text(24, 36, title, 20, INK, weight="700")
        y = 60
        for line in wrap(subtitle, width - 48, 7.0):
            self.text(24, y, line, 13, MUTED)
            y += 18
        self.top = y + 8

    # --- primitives

    def text(self, x, y, s, size=12, fill=INK, weight="400", family=SANS, anchor="start", italic=False, layer=None):
        style = ' font-style="italic"' if italic else ""
        (layer if layer is not None else self.parts).append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{style}>{escape(s)}</text>'
        )

    def rect(self, x, y, w, h, fill="#fff", stroke=LINE, rx=8, dash=None, width=1.2, layer=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        (layer if layer is not None else self.parts).append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>'
        )
        self.bottom = max(self.bottom, y + h)

    def badges(self, x, y, w, items) -> float:
        """draws a row of badges, wrapping, returns the height used"""
        cx, cy = x, y
        for kind, label in items:
            bw = len(label) * 6.1 + 16
            if cx + bw > x + w and cx > x:
                cx, cy = x, cy + 24
            bg, fg = BADGES[kind]
            self.rect(cx, cy, bw, 19, fill=bg, stroke=bg, rx=9.5)
            self.text(cx + 8, cy + 13.5, label, 11, fg, weight="600")
            cx += bw + 6
        return cy + 19 - y if items else 0

    def arrow(self, points, label=None, at=None, dash=False, anchor="start", color=FAINT, vertical=False):
        d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in points)
        da = ' stroke-dasharray="5 4"' if dash else ""
        self.overlay.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{da} marker-end="url(#head)"/>'
        )
        if label and vertical:
            # a label running up along a vertical lane, centred on (lx, ly)
            lx, ly = at
            tw = len(label) * 6.0
            self.overlay.append(
                f'<g transform="rotate(-90 {lx:.1f} {ly:.1f})"><rect x="{lx - tw / 2 - 4:.1f}" y="{ly - 8:.1f}" '
                f'width="{tw + 8:.1f}" height="15" fill="#ffffff"/></g>'
            )
            self.overlay.append(
                f'<text x="{lx:.1f}" y="{ly + 3.5:.1f}" transform="rotate(-90 {lx:.1f} {ly:.1f})" font-family="{SANS}" '
                f'font-size="11" fill="{MUTED}" text-anchor="middle" font-style="italic">{escape(label)}</text>'
            )
        elif label:
            lx, ly = at or points[0]
            lines = label.split("\n")
            for i, line in enumerate(lines):
                tw = len(line) * 6.0
                bx = lx - (tw / 2 if anchor == "middle" else tw if anchor == "end" else 0)
                self.overlay.append(
                    f'<rect x="{bx - 3:.1f}" y="{ly - 11 + i * 15:.1f}" width="{tw + 6:.1f}" height="15" fill="#ffffff"/>'
                )
                self.text(lx, ly + i * 15, line, 11, MUTED, italic=True, anchor=anchor, layer=self.overlay)

    # --- composites

    def card(self, x, y, w, name, when=None, then=None, badges=()) -> Box:
        """an operator: its name, WHEN it applies and THEN what it does, and badges for what it uses"""
        body: list[tuple[str, list[str]]] = []
        for label, text in (("WHEN", when), ("THEN", then)):
            if text:
                body.append((label, wrap(text, w - 70)))
        h = 30 + sum(len(lines) * 16 + 6 for _, lines in body) + 8
        bh = 0
        if badges:
            # measure the badge rows the same way they are drawn
            rows, cx = 1, 0
            for _, label in badges:
                bw = len(label) * 6.1 + 22
                if cx + bw > w - 20 and cx > 0:
                    rows, cx = rows + 1, 0
                cx += bw
            bh = rows * 24
        h += bh
        self.rect(x, y, w, h, fill="#f8fafc", stroke=LINE)
        self.parts.append(
            f'<path d="M {x} {y + 26} V {y + 8} a 8 8 0 0 1 8 -8 H {x + w - 8} a 8 8 0 0 1 8 8 V {y + 26} Z" '
            f'fill="#e2e8f0"/>'
        )
        self.text(x + 12, y + 18, name, 13, INK, weight="700", family=MONO)
        cy = y + 46
        for label, lines in body:
            self.text(x + 12, cy, label, 10, FAINT, weight="700")
            for line in lines:
                self.text(x + 58, cy, line, 12, INK)
                cy += 16
            cy += 6
        if badges:
            self.badges(x + 12, cy - 8, w - 24, badges)
        return Box(x, y, w, h)

    def panel(self, x, y, w, title, rows) -> Box:
        """a reference panel: a title and rows of (term, explanation)"""
        lines: list[tuple[str, str]] = []
        for term, expl in rows:
            lines.append(("mono", term))
            for line in wrap(expl, w - 32):
                lines.append(("sans", line))
        h = 34 + len(lines) * 16 + 10
        self.rect(x, y, w, h, fill="#ffffff", stroke="#cbd5e1", dash="4 3")
        self.text(x + 12, y + 22, title, 12, INK, weight="700")
        cy = y + 44
        for kind, line in lines:
            if kind == "mono":
                self.text(x + 12, cy, line, 12, "#0f172a", weight="600", family=MONO)
            else:
                self.text(x + 22, cy, line, 11.5, MUTED)
            cy += 16
        return Box(x, y, w, h)

    def pill(self, x, y, w, title, text) -> Box:
        lines = wrap(text, w - 24, 6.4)
        h = 32 + len(lines) * 15 + 6
        self.rect(x, y, w, h, fill=INK, stroke=INK, rx=14)
        self.text(x + 14, y + 20, title, 12, "#ffffff", weight="700")
        for i, line in enumerate(lines):
            self.text(x + 14, y + 38 + i * 15, line, 11.5, "#cbd5e1")
        return Box(x, y, w, h)

    def section(self, x, y, w, title, note=None):
        self.text(x, y, title.upper(), 11, FAINT, weight="700")
        self.parts.append(f'<line x1="{x}" y1="{y + 7}" x2="{x + w}" y2="{y + 7}" stroke="#e2e8f0" stroke-width="1"/>')
        if note:
            self.text(x + len(title) * 8.2 + 14, y, note, 11.5, MUTED, italic=True)

    def legend(self, y):
        x = 24
        self.text(x, y + 13.5, "Uses:", 11.5, MUTED, weight="600")
        x += 42
        for kind, label in (("llm", "language model"), ("world", "world graph / reasoner"), ("sub", "sub-process"),
                            ("stack", "intent stack"), ("turn", "ends the turn")):
            bg, fg = BADGES[kind]
            bw = len(label) * 6.1 + 16
            self.rect(x, y, bw, 19, fill=bg, stroke=bg, rx=9.5)
            self.text(x + 8, y + 13.5, label, 11, fg, weight="600")
            x += bw + 8
        self.text(x + 8, y + 13.5, "Arrows: what usually applies next, labelled with the state that lets it.",
                  11.5, MUTED, italic=True)

    def save(self, name: str):
        h = self.bottom + 28
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{h:.0f}" '
            f'viewBox="0 0 {self.w} {h:.0f}" role="img" aria-label="{escape(name)}">\n'
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{FAINT}"/></marker></defs>\n'
            f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
            + "\n".join(self.parts + self.overlay)
            + "\n</svg>\n"
        )
        (OUT / name).write_text(svg)


# --- the sub-processes: inputs and state on the left, operators in the middle, output and termination on the right

LEFT_X, LEFT_W = 24, 236
MID_X, MID_W = 292, 440
RIGHT_X, RIGHT_W = 812, 236
WIDTH = 1072


def filler_frame(f: Fig, inputs, state, output, done, called_by):
    y = f.top + 34
    f.legend(f.top)
    a = f.panel(LEFT_X, y, LEFT_W, "Inputs (io.a)", inputs)
    f.panel(LEFT_X, a.y + a.h + 16, LEFT_W, "State", state)
    o = f.panel(RIGHT_X, y, RIGHT_W, "Output", output)
    c = f.panel(RIGHT_X, o.y + o.h + 16, RIGHT_W, "Called by", called_by)
    return y, c.y + c.h + 16, done


def shipping_request_filler():
    f = Fig(WIDTH, "ShippingRequestFiller  ·  fill_shipping_request.py",
            "Fills in the parts of a shipping request missing so far from one utterance, read as the answer to the "
            "question the agent asked, and returns the request with an ask for each part still missing.")
    y, ry, done = filler_frame(
        f,
        [("utterance", "what the human said"), ("answering", "the ask it answers, or None"),
         ("reasoner", "the world graph, for the cities"), ("request", "the ShippingRequest so far")],
        [("request", "the copy being filled in"), ("to_fill", "the missing parts"),
         ("extracted", "whether the utterance has been used"), ("asks", "None until composed")],
        [("request", "the ShippingRequest"),
         ("asks", "AskFor(ShippingRequest, part) for each part still missing; empty when complete")],
        None,
        [("ReactToShippingRequestIntent", "in the conductor, with SHIPPING_REQUEST on top")],
    )
    t = f.card(MID_X, y, MID_W, "TakeRequest",
               "the request has not been taken",
               "copy the conductor's shipping request filled in so far, and note its missing parts")
    e = f.card(MID_X, t.y + t.h + 56, MID_W, "ExtractParts",
               "the request is taken, parts are missing, and the utterance has not been used",
               "fill in every missing part in one call: the origin and destination as city names, checked against "
               "the world graph, and the shipment (what, and how much it weighs). A part stays missing when the "
               "utterance does not give it, or names a city the carrier does not serve.",
               [("llm", "ModelPopulator (parts_model)"), ("world", "city_named")])
    c = f.card(MID_X, e.y + e.h + 56, MID_W, "ComposeAsks",
               "the utterance has been used, or no part was missing",
               "ask for each part still missing, in the order origin, destination, shipment; ask nothing when the "
               "request is complete",
               [("stack", "AskFor(ShippingRequest, part)")])
    d = f.pill(MID_X + 90, c.y + c.h + 40, MID_W - 180, "Done: asks_composed",
               "the asks are composed (empty when the request is complete)")
    f.arrow([t.bottom(0.4), e.top(0.4)], "parts missing", (t.bottom(0.4)[0] + 8, t.y + t.h + 32))
    f.arrow([e.bottom(0.4), c.top(0.4)], "extracted", (e.bottom(0.4)[0] + 8, e.y + e.h + 32))
    lane = MID_X + MID_W + 30
    f.arrow([t.right(0.5), (lane, t.right(0.5)[1]), (lane, c.right(0.5)[1]), c.right(0.5)],
            "nothing missing", (lane, (t.y + t.h / 2 + c.y + c.h / 2) / 2), dash=True, vertical=True)
    f.arrow([c.bottom(), d.top()])
    f.save("shipping-request-filler.svg")


def vehicle_filler():
    f = Fig(WIDTH, "VehicleFiller  ·  fill_vehicle.py",
            "Fills in a vehicle to add to the fleet: first its kind, which starts a draft of that model, then the "
            "draft's missing fields, all together. Returns the draft with an ask for what is still missing.")
    y, ry, _ = filler_frame(
        f,
        [("utterance", "what the human said"), ("answering", "the ask it answers, or None"),
         ("reasoner", "the world graph, for a fresh name"), ("draft", "the vehicle draft so far, or None")],
        [("taken", "whether the draft has been taken"), ("draft", "the copy being filled in, or None"),
         ("kind_tried", "whether the kind was classified"), ("to_fill", "the draft's missing fields"),
         ("extracted", "whether the fields were extracted"), ("asks", "None until composed")],
        [("draft", "a Truck or Airplane draft, or None when the kind is not known"),
         ("asks", "AskForKind(VehicleType) without a draft; otherwise AskFor(model, field) per missing field")],
        None,
        [("ReactToAddToFleetIntent", "in the conductor, with ADD_TO_FLEET on top, unless the human answers "
                                     "where the vehicle is")],
    )
    t = f.card(MID_X, y, MID_W, "TakeDraft",
               "the draft has not been taken",
               "copy the conductor's draft of the vehicle, if there is one, and note its missing fields")
    k = f.card(MID_X, t.y + t.h + 56, MID_W, "ChooseKind",
               "there is no draft, and the kind has not been tried on this utterance",
               "classify the kind among the VehicleTypes, truck or airplane, with the question asked as context; "
               "start a draft of its model, named with a fresh name such as truck3. No kind: still no draft.",
               [("llm", "EnumClassifier(VehicleType)"), ("world", "fresh_name")])
    e = f.card(MID_X, k.y + k.h + 56, MID_W, "ExtractFields",
               "the draft has missing fields, and the utterance has not been used for them",
               "fill in every missing field in one call; a field stays missing when the utterance does not give "
               "it, or gives blank text or a number not above zero",
               [("llm", "ModelPopulator (fields_model)")])
    c = f.card(MID_X, e.y + e.h + 56, MID_W, "ComposeAsks",
               "the utterance has been used: the kind tried, or the fields extracted",
               "with no draft, ask for the kind; otherwise ask for each field still missing, in the model's order; "
               "ask nothing when the vehicle is complete",
               [("stack", "AskForKind(VehicleType)"), ("stack", "AskFor(model, field)")])
    d = f.pill(MID_X + 90, c.y + c.h + 40, MID_W - 180, "Done: asks_composed",
               "the asks are composed (empty when the vehicle is complete)")
    f.arrow([t.bottom(0.4), k.top(0.4)], "no draft yet", (t.bottom(0.4)[0] + 8, t.y + t.h + 32))
    f.arrow([k.bottom(0.4), e.top(0.4)], "kind found: draft started", (k.bottom(0.4)[0] + 8, k.y + k.h + 32))
    f.arrow([e.bottom(0.4), c.top(0.4)], "extracted", (e.bottom(0.4)[0] + 8, e.y + e.h + 32))
    lane1 = MID_X + MID_W + 30
    f.arrow([t.right(0.5), (lane1, t.right(0.5)[1]), (lane1, e.right(0.3)[1]), e.right(0.3)],
            "a draft with missing fields", (lane1, (t.y + t.h / 2 + e.y + e.h * 0.3) / 2), dash=True, vertical=True)
    lx = MID_X - 22
    f.arrow([k.left(0.5), (lx, k.left(0.5)[1]), (lx, c.left(0.5)[1]), c.left(0.5)],
            "no kind: still no draft", (lx, (k.y + k.h / 2 + c.y + c.h / 2) / 2), dash=True, vertical=True)
    f.arrow([c.bottom(), d.top()])
    f.save("vehicle-filler.svg")


def placement_filler():
    f = Fig(WIDTH, "PlacementFiller  ·  fill_placement.py",
            "Fills in where a vehicle being added is: the location of its At relation, chosen among the places the "
            "world graph has that the vehicle can be at. Returns the draft with an ask while it is missing.")
    y, ry, _ = filler_frame(
        f,
        [("utterance", "what the human said"), ("answering", "the ask it answers, or None"),
         ("reasoner", "the world graph, for the places"), ("draft", "the At draft, the vehicle as entity1")],
        [("draft", "the copy being filled in"), ("to_fill", "the missing fields: entity2"),
         ("extracted", "whether the utterance has been used"), ("asks", "None until composed")],
        [("draft", "the At draft, with entity2 once the location is known"),
         ("asks", "AskFor(At, \"entity2\") while it is missing; empty once known")],
        None,
        [("ReactToAddToFleetIntent", "in the conductor, with ADD_TO_FLEET on top, when the human answers where "
                                     "the vehicle is")],
    )
    t = f.card(MID_X, y, MID_W, "TakeDraft",
               "the draft has not been taken",
               "copy the conductor's draft of the At relation, with the vehicle as entity1, and note what is missing")
    e = f.card(MID_X, t.y + t.h + 56, MID_W, "ExtractLocation",
               "the location (entity2) is missing, and the utterance has not been used",
               "fill in the location, choosing among the places the vehicle can be at: the airports for an airplane, "
               "any terminal or airport otherwise. Anywhere else, it stays missing.",
               [("llm", "ModelPopulator (location_model)"), ("world", "places_for")])
    c = f.card(MID_X, e.y + e.h + 56, MID_W, "ComposeAsks",
               "the utterance has been used, or nothing was missing",
               "ask \"Where is it now: which terminal or airport?\" while the location is missing; ask nothing once "
               "it is known",
               [("stack", "AskFor(At, \"entity2\")")])
    d = f.pill(MID_X + 90, c.y + c.h + 40, MID_W - 180, "Done: asks_composed",
               "the asks are composed (empty once the location is known)")
    f.arrow([t.bottom(0.4), e.top(0.4)], "location missing", (t.bottom(0.4)[0] + 8, t.y + t.h + 32))
    f.arrow([e.bottom(0.4), c.top(0.4)], "extracted", (e.bottom(0.4)[0] + 8, e.y + e.h + 32))
    lane = MID_X + MID_W + 30
    f.arrow([t.right(0.5), (lane, t.right(0.5)[1]), (lane, c.right(0.5)[1]), c.right(0.5)],
            "nothing missing", (lane, (t.y + t.h / 2 + c.y + c.h / 2) / 2), dash=True, vertical=True)
    f.arrow([c.bottom(), d.top()])
    f.save("placement-filler.svg")


def route_filler():
    f = Fig(WIDTH, "RouteFiller  ·  fill_route.py",
            "Fills in which route the human reports blocked: the cities at its ends, and whether it is the road or "
            "the flights, classified from what blocks it. Returns the report and kind with asks for what is missing.")
    y, ry, _ = filler_frame(
        f,
        [("utterance", "what the human said"), ("answering", "the ask it answers, or None"),
         ("reasoner", "the world graph, for the cities"), ("report", "the RouteReport so far"),
         ("kind", "the RouteType, or None")],
        [("report", "the copy being filled in"), ("kind", "the RouteType, or None"),
         ("to_fill", "the missing cities"), ("kind_tried", "whether the kind was classified"),
         ("extracted", "whether the cities were extracted"), ("asks", "None until composed")],
        [("report", "the RouteReport: city1, city2"), ("kind", "RouteType.ROAD or AIR, or None"),
         ("asks", "AskFor(RouteReport, city) per missing city, then AskForKind(RouteType)")],
        None,
        [("ReactToObstructRouteIntent", "in the conductor, with OBSTRUCT_ROUTE on top")],
    )
    t = f.card(MID_X, y, MID_W, "TakeReport",
               "the report has not been taken",
               "copy the conductor's report filled in so far and the kind of route, and note the missing cities")
    gap = 24
    hw = (MID_W - gap) / 2
    k = f.card(MID_X - 20, t.y + t.h + 70, 230, "ChooseRouteType",
               "the kind is not known, and not tried on this utterance",
               "classify the kind among the RouteTypes, from what blocks the route: snowed in means the road, "
               "flights grounded means the air. Unknown when the cause does not tell.",
               [("llm", "EnumClassifier(RouteType)")])
    e = f.card(MID_X + MID_W + 20 - 230, t.y + t.h + 70, 230, "ExtractCities",
               "cities are missing, and not extracted from this utterance",
               "fill in the missing cities in one call, checked against the world graph's cities; a city the "
               "carrier does not serve stays missing",
               [("llm", "ModelPopulator (cities_model)"), ("world", "city_named")])
    low = max(k.y + k.h, e.y + e.h)
    c = f.card(MID_X, low + 70, MID_W, "ComposeAsks",
               "the utterance has been used: the kind tried unless known, the cities extracted unless none missing",
               "ask for each city still missing, in order, then for the kind if it is still unknown; ask nothing once "
               "the route is known",
               [("stack", "AskFor(RouteReport, city)"), ("stack", "AskForKind(RouteType)")])
    d = f.pill(MID_X + 90, c.y + c.h + 40, MID_W - 180, "Done: asks_composed",
               "the asks are composed (empty once the route is known)")
    kx, ex = k.x + k.w / 2, e.x + e.w / 2
    f.arrow([(kx, t.y + t.h), (kx, k.y)], "kind unknown", (kx - 8, t.y + t.h + 40), anchor="end")
    f.arrow([(ex, t.y + t.h), (ex, e.y)], "cities missing", (ex + 8, t.y + t.h + 40))
    f.arrow([(kx, k.y + k.h), (kx, c.y)])
    f.arrow([(ex, e.y + e.h), (ex, c.y)])
    f.text(MID_X + MID_W / 2, t.y + t.h + 40, "either order", 11, MUTED, anchor="middle", italic=True)
    f.arrow([c.bottom(), d.top()])
    f.save("route-filler.svg")


def state_inspector():
    f = Fig(WIDTH, "StateInspector  ·  inspect_state.py",
            "Answers a question about the state of the world. The language model only understands the question; "
            "the facts come from the world graph, and the conductor words them with a fixed template.")
    y, ry, _ = filler_frame(
        f,
        [("utterance", "what the human asked"), ("answering", "the ask it answers, or None"),
         ("reasoner", "the world graph, which holds the state")],
        [("classified", "whether the question was classified"), ("question_type", "a QuestionType, or None"),
         ("fleet_question", "what it is about, or None"), ("sightings", "the facts, or None")],
        [("question_type", "FLEET, or None: not a question the agent answers"),
         ("sightings", "Sighting(vehicle, place) per vehicle asked about, in name order")],
        None,
        [("ReactToInspectStateIntent", "in the conductor, with INSPECT_STATE on top; it words the sightings "
                                       "with fleet_answer")],
    )
    k = f.card(MID_X, y, MID_W, "ClassifyQuestion",
               "the question has not been classified",
               "classify which kind of question it is among the QuestionTypes, with the question asked as context; "
               "None when it is none of them",
               [("llm", "EnumClassifier(QuestionType)")])
    q = f.card(MID_X, k.y + k.h + 56, MID_W, "FillFleetQuestion",
               "a FLEET question, and what it is about is not filled in",
               "fill in FleetQuestion.kind: truck, airplane, or None for the whole fleet; the whole fleet too when "
               "the language model fails, so nothing is ever asked back",
               [("llm", "ModelPopulator(FleetQuestion)")])
    l = f.card(MID_X, q.y + q.h + 56, MID_W, "LookUpFleet",
               "what the question is about is filled in, and the facts are not looked up",
               "find the vehicles of that kind, or every vehicle, in the order of their names, and where each is "
               "from its At relation (place_of); no At relation means an unknown place",
               [("world", "find_entities"), ("world", "place_of")])
    d = f.pill(MID_X + 90, l.y + l.h + 40, MID_W - 180, "Done: answered",
               "the question is classified as none the agent answers, or its facts are looked up")
    f.arrow([k.bottom(0.4), q.top(0.4)], "FLEET", (k.bottom(0.4)[0] + 8, k.y + k.h + 32))
    f.arrow([q.bottom(0.4), l.top(0.4)], "filled in", (q.bottom(0.4)[0] + 8, q.y + q.h + 32))
    f.arrow([l.bottom(), d.top()])
    lane = MID_X + MID_W + 30
    f.arrow([k.right(0.5), (lane, k.right(0.5)[1]), (lane, d.right(0.5)[1]), d.right(0.5)],
            "None: not a question it answers", (lane, (k.y + k.h / 2 + d.y + d.h / 2) / 2), dash=True,
            vertical=True)
    f.save("state-inspector.svg")


# --- the conductor: listening operators, then one row per thing that can be on top of the intent stack

def conductor():
    W = 1240
    f = Fig(W, "Conductor  ·  conductor.py",
            "The agent's main decision process. While it listens, operators read the human's utterance and "
            "RecognizeIntent pushes its intent onto the intent stack. While it works, only the top of the stack "
            "decides which operator applies. Operators keep applying until one ends the turn, and the agent "
            "listens again.")
    f.legend(f.top)
    y = f.top + 50

    # the turn loop, with an example stack
    f.section(24, y, W - 48, "A turn")
    y += 22
    lb = Box(24, y, 300, 108)
    f.rect(lb.x, lb.y, lb.w, lb.h, fill="#f1f5f9", stroke=LINE, rx=14)
    f.text(lb.x + 16, lb.y + 28, "Listening", 15, INK, weight="700")
    for i, line in enumerate(wrap("state.listening: waiting for the human; the next utterance is read by the "
                                  "listening operators", lb.w - 32)):
        f.text(lb.x + 16, lb.y + 50 + i * 16, line, 12, MUTED)
    wb = Box(560, y, 300, 108)
    f.rect(wb.x, wb.y, wb.w, wb.h, fill="#f1f5f9", stroke=LINE, rx=14)
    f.text(wb.x + 16, wb.y + 28, "Working", 15, INK, weight="700")
    for i, line in enumerate(wrap("the operator for what is on top of the intent stack applies, again and again, "
                                  "noting partial replies as it goes", wb.w - 32)):
        f.text(wb.x + 16, wb.y + 50 + i * 16, line, 12, MUTED)
    f.arrow([lb.right(0.3), wb.left(0.3)], "RecognizeIntent pushes the intent",
            ((lb.x + lb.w + wb.x) / 2, lb.y + lb.h * 0.3 - 8), anchor="middle")
    f.arrow([wb.left(0.75), lb.right(0.75)], "AskQuestion, AskToConfirmChange or",
            ((lb.x + lb.w + wb.x) / 2, lb.y + lb.h * 0.75 + 20), anchor="middle")
    f.text((lb.x + lb.w + wb.x) / 2, lb.y + lb.h * 0.75 + 35, "Reply ends the turn: say the replies",
           11, MUTED, italic=True, anchor="middle")
    # example stack
    sx, sw = 930, 286
    f.text(sx, y + 4, "Example stack (top first)", 11.5, MUTED, weight="600")
    items = ['AskFor(ShippingRequest, "destination")', "AskFor(ShippingRequest)", "ESTIMATE_PRICE"]
    for i, it in enumerate(items):
        top = i == 0
        f.rect(sx, y + 14 + i * 30, sw, 26, fill="#fce7f3" if top else "#ffffff",
               stroke="#be185d" if top else "#cbd5e1", rx=5)
        f.text(sx + 10, y + 31.5 + i * 30, it, 11.5, INK, family=MONO, weight="600" if top else "400")
    f.text(sx, y + 120, "the top asks for the destination: AskQuestion", 11.5, MUTED, italic=True)
    y += 140

    # listening operators
    f.section(24, y, W - 48, "Listening", "the utterance picks the operator")
    y += 20
    cw = (W - 48 - 3 * 14) / 4
    cards = [
        ("Introduce", "the agent has not introduced itself", "introduce the agent and what it can help with",
         [("turn", "ends turn")]),
        ("SayGoodbye", "the utterance contains a quit phrase", "say goodbye and end the conversation",
         [("turn", "ends conversation")]),
        ("Acknowledge", "the utterance only acknowledges, such as \"thanks\" or \"ok\" (not while confirming)",
         "reply to thanks and ask the last question again, if any; otherwise say nothing", [("turn", "ends turn")]),
        ("RecognizeIntent", "anything else, or an acknowledgement while confirming",
         "recognize the HumanIntent; push it, or NotUnderstood, and start working",
         [("llm", "EnumClassifier(HumanIntent)"), ("stack", "push")]),
    ]
    boxes = [f.card(24 + i * (cw + 14), y, cw, *c) for i, c in enumerate(cards)]
    y = max(b.y + b.h for b in boxes) + 36

    # working operators, one row per top of the stack
    f.section(24, y, W - 48, "Working", "the top of the intent stack picks the operator")
    y += 18
    TOP_X, TOP_W = 24, 300
    OP_X = TOP_X + TOP_W + 36
    OP_W = W - 24 - OP_X
    groups = [
        ("Conversation", [
            ("GREETING", None, "ReactToGreetingIntent", "note a friendly reply; pop", [("stack", "pop")]),
            ("ASK_CAPABILITIES", None, "ReactToAskCapabilitiesIntent", "note what the agent can help with; pop",
             [("stack", "pop")]),
            ("NotUnderstood", "no known intent matched", "RespondUnknown",
             "note that the agent is not designed to respond to that; pop, leaving the rest of the stack",
             [("stack", "pop")]),
        ]),
        ("Shipping requests and prices", [
            ("SHIPPING_REQUEST", None, "ReactToShippingRequestIntent",
             "fill in the request being asked for, or a new one; pop; push AskFor(ShippingRequest) with an ask per "
             "missing part above it",
             [("sub", "ShippingRequestFiller"), ("stack", "pop, push asks")]),
            ("AskFor(ShippingRequest)", "the request is complete", "ConfirmShippingRequest",
             "note the delivery as understood, inviting the price question unless it was asked; pop",
             [("stack", "pop")]),
            ("ESTIMATE_PRICE", "the request is not complete", "ReactToEstimatePriceIntent",
             "push AskFor(ShippingRequest) with an ask per missing part; the intent waits beneath them",
             [("stack", "push asks")]),
            ("ESTIMATE_PRICE", "complete, not yet planned", "PlanDelivery",
             "plan the cheapest delivery from the origin to the destination; record the plan and its price",
             [("world", "planner search")]),
            ("ESTIMATE_PRICE", "the delivery is planned", "DescribePlan",
             "note the price and the plan's steps, or that no delivery could be planned; pop",
             [("llm", "describe_facts"), ("stack", "pop")]),
        ]),
        ("Adding to the fleet", [
            ("ADD_TO_FLEET", None, "ReactToAddToFleetIntent",
             "fill in where the vehicle is if the utterance answers that, else the vehicle, being asked for or "
             "new; pop; push AskFor(vehicle or At) with asks above it",
             [("sub", "VehicleFiller"), ("sub", "PlacementFiller"), ("stack", "pop, push asks")]),
            ("AskFor(Truck | Airplane)", "the vehicle is complete", "PlaceVehicle",
             "build the vehicle; start a draft of its At relation; pop; push AskFor(At) and its ask for where",
             [("stack", "pop, push asks")]),
            ("AskFor(At)", "the location is known", "ProposeAddition",
             "pop; keep a VehicleAddition as the pending change; push ConfirmChange",
             [("stack", "pop, push ConfirmChange")]),
        ]),
        ("Closing routes", [
            ("OBSTRUCT_ROUTE", None, "ReactToObstructRouteIntent",
             "fill in the route being asked for, or a new one; pop; push AskFor(RouteReport) with asks above it",
             [("sub", "RouteFiller"), ("stack", "pop, push asks")]),
            ("AskFor(RouteReport)", "both cities and the kind known", "ProposeClosure",
             "pop; if the world graph has such a route, keep a RouteClosure as the pending change and push "
             "ConfirmChange; otherwise note there is none",
             [("world", "blocked_route"), ("stack", "pop, push ConfirmChange")]),
        ]),
        ("Questions about the state", [
            ("INSPECT_STATE", None, "ReactToInspectStateIntent",
             "answer from the world graph, worded by fleet_answer, or say the agent can only answer about the "
             "fleet; pop",
             [("sub", "StateInspector"), ("stack", "pop")]),
        ]),
        ("Confirming changes", [
            ("ConfirmChange", None, "AskToConfirmChange",
             "say the replies and ask whether to make the pending change, e.g. \"Shall I close the road between "
             "Seattle and San Francisco?\"; it stays on the stack",
             [("turn", "ends turn")]),
            ("CONFIRM", "the human said yes", "ReactToConfirmIntent",
             "pop; if a change was being confirmed, make it, note what was done, pop ConfirmChange, and forget it",
             [("world", "add_entity / remove_relation"), ("stack", "pop")]),
            ("DENY", "the human said no", "ReactToDenyIntent",
             "pop; if a change was being confirmed, note it was not made, pop ConfirmChange, and forget it",
             [("stack", "pop")]),
        ]),
        ("Asks, for any model", [
            ("AskFor(model, attribute)\nAskForKind(enum)", "the agent has what it asked for", "PopSatisfiedAsk",
             "pop; the ask below is looked at next, so the asks cascade down to the whole-model ask and its "
             "operator: ConfirmShippingRequest, PlaceVehicle, ProposeAddition or ProposeClosure",
             [("stack", "pop")]),
            ("AskFor(…) or AskForKind(…)", "the agent does not have it yet", "AskQuestion",
             "say the replies and the ask's question, from the model's docstring, a Field description or the "
             "enum's docstring; the ask stays",
             [("turn", "ends turn")]),
        ]),
        ("Nothing left", [
            ("(empty stack)", None, "Reply", "say the replies noted this turn", [("turn", "ends turn")]),
        ]),
    ]
    for gname, rows in groups:
        y += 14
        f.text(24, y + 4, gname, 13, INK, weight="700")
        y += 14
        for top, cond, op, then, badges in rows:
            # the stack item on top, and the condition that picks this operator
            tops = top.split("\n")
            cond_lines = wrap(cond, TOP_W - 20) if cond else []
            th = 14 + len(tops) * 17 + len(cond_lines) * 15 + 8
            # the operator
            lines = wrap(then, OP_W - 300 - 20)
            oh = max(36, 12 + len(lines) * 16 + 4)
            bh = 0
            # badges under the operator's name
            rowsb, cx = 1, 0
            for _, label in badges:
                bw = len(label) * 6.1 + 22
                if cx + bw > 280 and cx > 0:
                    rowsb, cx = rowsb + 1, 0
                cx += bw
            bh = rowsb * 24
            oh = max(oh, 30 + bh + 4)
            h = max(th, oh)
            f.rect(TOP_X, y, TOP_W, h, fill="#ffffff", stroke="#e9a8c9", rx=6)
            for i, line in enumerate(tops):
                f.text(TOP_X + 12, y + 20 + i * 17, line, 12, "#9d174d", weight="600", family=MONO)
            for i, line in enumerate(cond_lines):
                f.text(TOP_X + 12, y + 20 + len(tops) * 17 + i * 15 - 2, line, 11.5, MUTED, italic=True)
            f.arrow([(TOP_X + TOP_W + 2, y + h / 2), (OP_X - 2, y + h / 2)])
            f.rect(OP_X, y, OP_W, h, fill="#f8fafc", stroke=LINE, rx=6)
            f.text(OP_X + 12, y + 20, op, 12.5, INK, weight="700", family=MONO)
            f.badges(OP_X + 12, y + 28, 276, badges)
            for i, line in enumerate(lines):
                f.text(OP_X + 300, y + 20 + i * 16, line, 12, INK)
            y += h + 8
    f.save("conductor.svg")


shipping_request_filler()
vehicle_filler()
placement_filler()
route_filler()
state_inspector()
conductor()
