import { useState, useRef, useEffect } from "react"

const POPUP_WIDTH = 280

function NutrientRow({ label, value, unit }) {
    if (value == null) return null
    return (
        <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid var(--border)", fontSize: 13 }}>
            <span style={{ color: "var(--text)" }}>{label}</span>
            <span style={{ color: "var(--text-h)", fontFamily: "var(--mono)" }}>{value} {unit}</span>
        </div>
    )
}

function Popup({ entity, anchorRect, containerRect, onSelectCandidate }) {
    const [selected, setSelected] = useState(0)
    const match = entity.candidates[selected]

    const left = Math.min(
        anchorRect.left - containerRect.left,
        containerRect.width - POPUP_WIDTH - 8
    )
    const top = anchorRect.bottom - containerRect.top + 8

    return (
        <div style={{
            position: "absolute", left, top,
            width: POPUP_WIDTH,
            background: "var(--bg)",
            border: "1px solid var(--accent-border)",
            borderRadius: 8,
            boxShadow: "var(--shadow)",
            padding: 12,
            zIndex: 100,
            textAlign: "left",
        }}>
            <div style={{ fontSize: 11, color: "var(--text)", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>
                {entity.unit ? `${entity.quantity} ${entity.unit} · ` : ""}
                {entity.grams != null ? `${entity.grams}g · ` : ""}
                {entity.candidates.length} candidate{entity.candidates.length !== 1 ? "s" : ""}
            </div>

            {/* Candidate selector */}
            <div style={{ display: "flex", flexDirection: "column", gap: 3, marginBottom: 10 }}>
                {entity.candidates.map((c, i) => (
                    <button key={c.food_id} onClick={() => { setSelected(i); onSelectCandidate?.(i) }}
                            style={{
                                textAlign: "left", background: i === selected ? "var(--accent-bg)" : "transparent",
                                border: i === selected ? "1px solid var(--accent-border)" : "1px solid transparent",
                                borderRadius: 5, padding: "4px 8px", cursor: "pointer",
                                color: i === selected ? "var(--accent)" : "var(--text)",
                                fontSize: 12, lineHeight: "140%",
                            }}>
                        {c.name}
                    </button>
                ))}
            </div>

            {/* Ingredients for recipes */}
            {match && match.is_recipe && match.recipe_ingredients && match.recipe_ingredients.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: "var(--text)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Ingredients</div>
                    {match.recipe_ingredients.map(ing => (
                        <div key={ing.food_id} style={{ fontSize: 12, color: "var(--text)", marginBottom: 2 }}>
                            {ing.name} ({(ing.weight_fraction * 100).toFixed(1)}%)
                        </div>
                    ))}
                </div>
            )}

            {/* Nutrients */}
            {match && (
                <div>
                    <div style={{ fontSize: 11, color: "var(--text)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Nutrition</div>
                    <NutrientRow label="Energy" value={match.nutrients.energy_kj} unit="kJ" />
                    <NutrientRow label="Protein" value={match.nutrients.protein_g} unit="g" />
                    <NutrientRow label="Fat" value={match.nutrients.fat_g} unit="g" />
                    <NutrientRow label="Carbs" value={match.nutrients.carbs_g} unit="g" />
                    <NutrientRow label="Fibre" value={match.nutrients.fibre_g} unit="g" />
                    <NutrientRow label="Sodium" value={match.nutrients.sodium_mg} unit="mg" />
                </div>
            )}
        </div>
    )
}

function AnnotatedText({ text, entities, onEntityClick, activeEntity }) {
    if (!text || entities.length === 0) return <span style={{ color: "var(--text)" }}>{text}</span>

    // Build a flat list of annotated regions from entity char spans
    const regions = [] // {start, end, type: "qty"|"unit"|"food", entityIndex}
    entities.forEach((ent, i) => {
        if (ent.quantity_char_start != null)
            regions.push({ start: ent.quantity_char_start, end: ent.quantity_char_end, type: "qty", i })
        if (ent.unit_char_start != null)
            regions.push({ start: ent.unit_char_start, end: ent.unit_char_start + (ent.unit ? ent.unit.length : 0), type: "unit", i })
        regions.push({ start: ent.char_start, end: ent.char_end, type: "food", i })
    })
    regions.sort((a, b) => a.start - b.start)

    const parts = []
    let cursor = 0
    for (const r of regions) {
        if (r.start > cursor)
            parts.push({ text: text.slice(cursor, r.start), type: "plain" })
        parts.push({ text: text.slice(r.start, r.end), type: r.type, i: r.i })
        cursor = r.end
    }
    if (cursor < text.length)
        parts.push({ text: text.slice(cursor), type: "plain" })

    const styles = {
        plain: { color: "var(--text)" },
        qty: { background: "var(--code-bg)", color: "var(--text-h)", borderRadius: 3, padding: "1px 4px", fontFamily: "var(--mono)", fontSize: "0.9em" },
        unit: { background: "var(--code-bg)", color: "var(--text-h)", borderRadius: 3, padding: "1px 4px", fontFamily: "var(--mono)", fontSize: "0.9em" },
        food: {
            background: "var(--accent-bg)", color: "var(--accent)",
            borderRadius: 4, padding: "1px 5px",
            borderBottom: "2px solid var(--accent-border)",
            cursor: "pointer", fontWeight: 500,
        },
    }

    return (
        <span>
            {parts.map((p, idx) => (
                <span key={idx}
                      style={{
                          ...styles[p.type],
                          ...(p.type === "food" && p.i === activeEntity ? { outline: "2px solid var(--accent)" } : {})
                      }}
                      onClick={p.type === "food" ? (e) => onEntityClick(p.i, e) : undefined}>
                    {p.text}
                </span>
            ))}
        </span>
    )
}

export default function App() {
    const [input, setInput] = useState("")
    const [result, setResult] = useState(null)
    const [activeEntity, setActiveEntity] = useState(null)
    const [anchorRect, setAnchorRect] = useState(null)
    const containerRef = useRef(null)

    const handleExtract = () => {
        fetch("http://localhost:8000/extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: input }),
        })
            .then(r => r.json())
            .then(json => { setResult(json); setActiveEntity(null) })
            .catch(console.error)
    }

    const handleEntityClick = (i, e) => {
        if (activeEntity === i) { setActiveEntity(null); return }
        setActiveEntity(i)
        setAnchorRect(e.currentTarget.getBoundingClientRect())
    }

    // Close popup on outside click
    useEffect(() => {
        const handler = (e) => {
            if (!containerRef.current?.contains(e.target)) setActiveEntity(null)
        }
        document.addEventListener("mousedown", handler)
        return () => document.removeEventListener("mousedown", handler)
    }, [])

    return (
        <div style={{ padding: "40px 48px", textAlign: "left", maxWidth: 720, margin: "0 auto" }}>
            <h1 style={{ marginBottom: 8, paddingBottom: 16 }}>Natural language input</h1>

            <div style={{ display: "flex", gap: 8, marginBottom: 32 }}>
                <input
                    style={{
                        flex: 1, padding: "10px 14px", borderRadius: 7,
                        border: "1px solid var(--border)", background: "var(--code-bg)",
                        color: "var(--text-h)", font: "inherit", fontSize: 16,
                        outline: "none",
                    }}
                    placeholder="e.g. two eggs and a slice of toast with butter…"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && handleExtract()}
                />
                <button
                    onClick={handleExtract}
                    style={{
                        padding: "10px 20px", borderRadius: 7, border: "none",
                        background: "var(--accent)", color: "#fff", cursor: "pointer",
                        fontFamily: "var(--sans)", fontSize: 15, fontWeight: 500,
                    }}>
                    Extract
                </button>
            </div>

            {result && (
                <div ref={containerRef} style={{ position: "relative" }}>
                    <h2 style={{ marginBottom: 12 }}>Result</h2>

                    {/* Annotated input string */}
                    <div style={{
                        fontSize: 20, lineHeight: "160%", marginBottom: 24,
                        padding: "16px 20px", borderRadius: 8,
                        border: "1px solid var(--border)", background: "var(--code-bg)",
                    }}>
                        <AnnotatedText
                            text={result.text}
                            entities={result.entities}
                            onEntityClick={handleEntityClick}
                            activeEntity={activeEntity}
                        />
                    </div>

                    {/* Popup for active entity */}
                    {activeEntity !== null && anchorRect && containerRef.current && (
                        <Popup
                            entity={result.entities[activeEntity]}
                            anchorRect={anchorRect}
                            containerRect={containerRef.current.getBoundingClientRect()}
                            onSelectCandidate={() => {}}
                        />
                    )}

                    {/* Summary row */}
                    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                        {result.entities.map((ent, i) => (
                            <div key={i}
                                 onClick={(e) => handleEntityClick(i, e)}
                                 style={{
                                     padding: "8px 14px", borderRadius: 7, cursor: "pointer",
                                     border: `1px solid ${i === activeEntity ? "var(--accent)" : "var(--border)"}`,
                                     background: i === activeEntity ? "var(--accent-bg)" : "var(--code-bg)",
                                 }}>
                                <div style={{ color: "var(--accent)", fontWeight: 500, fontSize: 14 }}>{ent.text}</div>
                                <div style={{ color: "var(--text)", fontSize: 12 }}>
                                    {ent.unit ? `${ent.quantity} ${ent.unit} (${ent.grams}g)` : ent.quantity !== 1 ? `${ent.quantity} × ${ent.grams}g` : ent.grams != null ? `${ent.grams}g` : "1 serving"}
                                </div>
                                {ent.match && (
                                    <div style={{ color: "var(--text)", fontSize: 11, marginTop: 2 }}>
                                        {ent.match.nutrients.energy_kj} kJ
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}