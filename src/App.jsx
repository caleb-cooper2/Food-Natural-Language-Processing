/*
 * Note: Claude was used to generate the frontend in this directory. The UI has been reviewed and integrated against the project's API.
 */

import { useState, useRef, useEffect, useCallback } from "react"

const API_URL = import.meta.env.VITE_NLP_API_URL ?? "http://localhost:8000"
const POPUP_WIDTH = 280
const DAILY_VALUES = { energy_kj: 8700, protein_g: 50, fat_g: 70, carbs_g: 310, fibre_g: 30, sodium_mg: 2000 }
const NUTRIENT_LABELS = [
    { key: "energy_kj", label: "Energy",  unit: "kJ", color: "#378ADD" },
    { key: "protein_g", label: "Protein", unit: "g",  color: "#1D9E75" },
    { key: "fat_g",     label: "Fat",     unit: "g",  color: "#D85A30" },
    { key: "carbs_g",   label: "Carbs",   unit: "g",  color: "#BA7517" },
    { key: "fibre_g",   label: "Fibre",   unit: "g",  color: "#639922" },
    { key: "sodium_mg", label: "Sodium",  unit: "mg", color: "#7F77DD" },
]

// Presentation helpers

function sumNutrients(entities, selectedCandidates) {
    const totals = { energy_kj: 0, protein_g: 0, fat_g: 0, carbs_g: 0, fibre_g: 0, sodium_mg: 0 }
    let hasAny = false
    entities.forEach((ent, i) => {
        const cidx = selectedCandidates[i] ?? 0
        const match = ent.candidates?.[cidx]
        if (!match?.nutrients) return
        Object.keys(totals).forEach(k => {
            const v = match.nutrients[k]
            if (v != null) { totals[k] = +(totals[k] + v).toFixed(2); hasAny = true }
        })
    })
    return hasAny ? totals : null
}

function NutrientSummary({ entities, selectedCandidates }) {
    const totals = sumNutrients(entities, selectedCandidates)
    if (!totals) return null
    return (
        <div style={{ marginTop: 28, borderTop: "1px solid var(--border)", paddingTop: 20 }}>
            <div style={{ fontSize: 11, color: "var(--text)", marginBottom: 14, textTransform: "uppercase", letterSpacing: 1 }}>
                Total nutrients
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 10, marginBottom: 20 }}>
                {NUTRIENT_LABELS.map(({ key, label, unit }) => {
                    const v = totals[key]
                    if (v == null) return null
                    return (
                        <div key={key} style={{ background: "var(--code-bg)", border: "1px solid var(--border)", borderRadius: 7, padding: "10px 14px" }}>
                            <div style={{ fontSize: 11, color: "var(--text)", marginBottom: 4 }}>{label}</div>
                            <div style={{ fontSize: 18, fontWeight: 500, color: "var(--text-h)" }}>
                                {v % 1 === 0 ? v : v.toFixed(1)}
                                <span style={{ fontSize: 12, fontWeight: 400, color: "var(--text)", marginLeft: 3 }}>{unit}</span>
                            </div>
                        </div>
                    )
                })}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {NUTRIENT_LABELS.map(({ key, label, color }) => {
                    const v = totals[key]
                    if (v == null) return null
                    const pct = Math.min(100, Math.round((v / DAILY_VALUES[key]) * 100))
                    return (
                        <div key={key} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <div style={{ width: 52, fontSize: 12, color: "var(--text)", textAlign: "right", flexShrink: 0 }}>{label}</div>
                            <div style={{ flex: 1, height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                                <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3, transition: "width 0.3s ease" }} />
                            </div>
                            <div style={{ width: 42, fontSize: 12, color: "var(--text)", flexShrink: 0 }}>{pct}% DV</div>
                        </div>
                    )
                })}
            </div>
            <div style={{ fontSize: 11, color: "var(--text)", marginTop: 8 }}>% of estimated daily values</div>
        </div>
    )
}

function NutrientRow({ label, value, unit }) {
    if (value == null) return null
    return (
        <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid var(--border)", fontSize: 13 }}>
            <span style={{ color: "var(--text)" }}>{label}</span>
            <span style={{ color: "var(--text-h)", fontFamily: "var(--mono)" }}>
                {typeof value === "number" ? (value % 1 === 0 ? value : value.toFixed(1)) : value} {unit}
            </span>
        </div>
    )
}

function Popup({ entity, anchorRect, containerRect, selectedIdx, onSelectCandidate }) {
    const match = entity.candidates[selectedIdx]
    const left = Math.min(anchorRect.left - containerRect.left, containerRect.width - POPUP_WIDTH - 8)
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
            <div style={{ display: "flex", flexDirection: "column", gap: 3, marginBottom: 10 }}>
                {entity.candidates.map((c, i) => (
                    <button key={c.food_id}
                            onClick={() => onSelectCandidate(i)}
                            style={{
                                textAlign: "left",
                                background: i === selectedIdx ? "var(--accent-bg)" : "transparent",
                                border: i === selectedIdx ? "1px solid var(--accent-border)" : "1px solid transparent",
                                borderRadius: 5, padding: "4px 8px", cursor: "pointer",
                                color: i === selectedIdx ? "var(--accent)" : "var(--text)",
                                fontSize: 12, lineHeight: "140%",
                                display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6,
                            }}>
                        <span style={{ flex: 1 }}>{c.name}</span>
                        <span style={{ fontSize: 11, opacity: 0.6, flexShrink: 0 }}>{c.score?.toFixed(0)}</span>
                    </button>
                ))}
            </div>
            {match && match.is_recipe && match.recipe_ingredients?.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: "var(--text)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Ingredients</div>
                    {match.recipe_ingredients.map(ing => (
                        <div key={ing.food_id} style={{ fontSize: 12, color: "var(--text)", marginBottom: 2 }}>
                            {ing.name} ({(ing.weight_fraction * 100).toFixed(1)}%)
                        </div>
                    ))}
                </div>
            )}
            {match && (
                <div>
                    <div style={{ fontSize: 11, color: "var(--text)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
                        Nutrition · {entity.grams}g
                    </div>
                    <NutrientRow label="Energy"  value={match.nutrients.energy_kj}  unit="kJ" />
                    <NutrientRow label="Protein" value={match.nutrients.protein_g}  unit="g" />
                    <NutrientRow label="Fat"     value={match.nutrients.fat_g}      unit="g" />
                    <NutrientRow label="Carbs"   value={match.nutrients.carbs_g}    unit="g" />
                    <NutrientRow label="Fibre"   value={match.nutrients.fibre_g}    unit="g" />
                    <NutrientRow label="Sodium"  value={match.nutrients.sodium_mg}  unit="mg" />
                </div>
            )}
        </div>
    )
}

function AnnotatedText({ text, entities, onEntityClick, activeEntity, selectedCandidates }) {
    if (!text || entities.length === 0) return <span style={{ color: "var(--text)" }}>{text}</span>

    const regions = []
    entities.forEach((ent, i) => {
        if (ent.quantity_char_start != null)
            regions.push({ start: ent.quantity_char_start, end: ent.quantity_char_end, type: "qty", i })
        if (ent.unit_char_start != null)
            regions.push({ start: ent.unit_char_start, end: ent.unit_char_start + (ent.unit ? ent.unit.length : 0), type: "unit", i })
        if (ent.char_start != null && ent.char_end != null)
            regions.push({ start: ent.char_start, end: ent.char_end, type: "food", i })
    })
    regions.sort((a, b) => a.start - b.start)

    const parts = []
    let cursor = 0
    for (const r of regions) {
        if (r.start < cursor) continue
        if (r.start > cursor) parts.push({ text: text.slice(cursor, r.start), type: "plain" })
        parts.push({ text: text.slice(r.start, r.end), type: r.type, i: r.i })
        cursor = r.end
    }
    if (cursor < text.length) parts.push({ text: text.slice(cursor), type: "plain" })

    const styles = {
        plain: { color: "var(--text)" },
        qty:  { background: "var(--code-bg)", color: "var(--text-h)", borderRadius: 3, padding: "1px 4px", fontFamily: "var(--mono)", fontSize: "0.9em" },
        unit: { background: "var(--code-bg)", color: "var(--text-h)", borderRadius: 3, padding: "1px 4px", fontFamily: "var(--mono)", fontSize: "0.9em" },
        food: { background: "var(--accent-bg)", color: "var(--accent)", borderRadius: 4, padding: "1px 5px", borderBottom: "2px solid var(--accent-border)", cursor: "pointer", fontWeight: 500 },
    }

    return (
        <span>
            {parts.map((p, idx) => (
                <span key={idx}
                      style={{
                          ...styles[p.type],
                          ...(p.type === "food" && p.i === activeEntity ? { outline: "2px solid var(--accent)" } : {}),
                          ...(p.type === "food" && (selectedCandidates[p.i] ?? 0) > 0 ? { textDecoration: "underline dotted" } : {}),
                      }}
                      onClick={p.type === "food" ? (e) => onEntityClick(p.i, e) : undefined}>
                    {p.text}
                </span>
            ))}
        </span>
    )
}

// ─── ResultView: shared annotated + JSON tab view ────────────────────────────

function ResultView({ result }) {
    const [tab, setTab] = useState("annotated")
    const [activeEntity, setActiveEntity] = useState(null)
    const [anchorRect, setAnchorRect] = useState(null)
    const [containerRect, setContainerRect] = useState(null)
    const [selectedCandidates, setSelectedCandidates] = useState({})
    const containerRef = useRef(null)

    useEffect(() => {
        const handler = (e) => {
            if (!containerRef.current?.contains(e.target)) setActiveEntity(null)
        }
        document.addEventListener("mousedown", handler)
        return () => document.removeEventListener("mousedown", handler)
    }, [])

    const handleEntityClick = (i, e) => {
        if (activeEntity === i) {
            setActiveEntity(null)
            setContainerRect(null)
            return
        }
        setActiveEntity(i)
        setAnchorRect(e.currentTarget.getBoundingClientRect())
        setContainerRect(containerRef.current?.getBoundingClientRect() ?? null)
    }

    const TAB_STYLE = (active) => ({
        padding: "5px 14px",
        fontSize: 12,
        fontFamily: "var(--mono)",
        border: "1px solid var(--border)",
        borderRadius: 5,
        background: active ? "var(--accent-bg)" : "transparent",
        color: active ? "var(--accent)" : "var(--text)",
        cursor: "pointer",
    })

    return (
        <div ref={containerRef} style={{ position: "relative" }}>
            {/* header bar */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <div style={{ display: "flex", gap: 6 }}>
                    <button style={TAB_STYLE(tab === "annotated")} onClick={() => setTab("annotated")}>annotated</button>
                    <button style={TAB_STYLE(tab === "json")} onClick={() => setTab("json")}>json</button>
                </div>
                <span style={{ fontSize: 11, color: "var(--text)", background: "var(--code-bg)", border: "1px solid var(--border)", borderRadius: 20, padding: "2px 10px" }}>
                    via {result.source}
                </span>
            </div>

            {tab === "annotated" && (
                <>
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
                            selectedCandidates={selectedCandidates}
                        />
                    </div>

                    {activeEntity !== null && anchorRect && containerRect && (
                        <Popup
                            entity={result.entities[activeEntity]}
                            anchorRect={anchorRect}
                            containerRect={containerRect}
                            selectedIdx={selectedCandidates[activeEntity] ?? 0}
                            onSelectCandidate={(cidx) => setSelectedCandidates(prev => ({ ...prev, [activeEntity]: cidx }))}
                        />
                    )}

                    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                        {result.entities.map((ent, i) => {
                            const cidx = selectedCandidates[i] ?? 0
                            const match = ent.candidates?.[cidx]
                            const isActive = i === activeEntity
                            const isOverridden = cidx > 0
                            return (
                                <div key={i}
                                     onClick={(e) => handleEntityClick(i, e)}
                                     style={{
                                         padding: "8px 14px", borderRadius: 7, cursor: "pointer",
                                         border: `1px solid ${isActive ? "var(--accent)" : "var(--border)"}`,
                                         background: isActive ? "var(--accent-bg)" : "var(--code-bg)",
                                     }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 1 }}>
                                        <span style={{ color: "var(--accent)", fontWeight: 500, fontSize: 14 }}>{ent.text}</span>
                                        {isOverridden && (
                                            <span style={{ fontSize: 10, color: "var(--text)", background: "var(--border)", borderRadius: 10, padding: "1px 6px" }}>edited</span>
                                        )}
                                    </div>
                                    <div style={{ color: "var(--text)", fontSize: 12 }}>
                                        {ent.unit ? `${ent.quantity} ${ent.unit} (${ent.grams}g)` : ent.quantity !== 1 ? `${ent.quantity} × ${ent.grams}g` : ent.grams != null ? `${ent.grams}g` : "1 serving"}
                                    </div>
                                    {match && (
                                        <div style={{ color: "var(--text)", fontSize: 11, marginTop: 2, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                            {match.name}
                                        </div>
                                    )}
                                    {match?.nutrients?.energy_kj != null && (
                                        <div style={{ color: "var(--text)", fontSize: 11, marginTop: 1 }}>
                                            {match.nutrients.energy_kj} kJ
                                        </div>
                                    )}
                                </div>
                            )
                        })}
                    </div>

                    <NutrientSummary entities={result.entities} selectedCandidates={selectedCandidates} />
                </>
            )}

            {tab === "json" && (
                <div style={{
                    background: "var(--code-bg)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    padding: "16px 20px",
                    overflow: "auto",
                    maxHeight: 520,
                }}>
                    <pre style={{
                        margin: 0,
                        fontSize: 12,
                        lineHeight: "170%",
                        fontFamily: "var(--mono)",
                        color: "var(--text-h)",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-all",
                    }}>
                        {JSON.stringify(result, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    )
}

// ─── BulkTestPanel ────────────────────────────────────────────────────────────

function BulkTestPanel() {
    const [rawInput, setRawInput]   = useState("")
    const [sentences, setSentences] = useState([])   // parsed list
    const [results, setResults]     = useState([])   // parallel array, null = pending/not-yet-run
    const [cursor, setCursor]       = useState(0)    // which sentence is currently shown
    const [running, setRunning]     = useState(false)
    const [done, setDone]           = useState(false)
    const abortRef = useRef(false)

    const parsed = rawInput.split("\n").map(s => s.trim()).filter(Boolean)

    const handleRun = useCallback(async () => {
        const lines = rawInput.split("\n").map(s => s.trim()).filter(Boolean)
        if (!lines.length) return

        abortRef.current = false
        setSentences(lines)
        setResults(Array(lines.length).fill(null))
        setCursor(0)
        setRunning(true)
        setDone(false)

        const acc = Array(lines.length).fill(null)

        for (let i = 0; i < lines.length; i++) {
            if (abortRef.current) break
            try {
                const resp = await fetch(`${API_URL}/extract`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: lines[i] }),
                })
                if (!resp.ok) throw new Error(`The API returned ${resp.status}`)
                acc[i] = await resp.json()
            } catch (err) {
                acc[i] = { error: String(err), text: lines[i], entities: [], source: "error" }
            }
            // spread so React sees a new array reference on each update
            setResults([...acc])
        }

        setRunning(false)
        setDone(true)
    }, [rawInput])

    const handleStop = () => { abortRef.current = true }

    const completedCount = results.filter(r => r !== null).length
    const progress = sentences.length ? completedCount / sentences.length : 0

    const canPrev = cursor > 0
    const canNext = cursor < sentences.length - 1

    const PILL = (active) => ({
        padding: "5px 16px", borderRadius: 20, fontSize: 12, fontWeight: 500,
        border: "1px solid var(--border)", cursor: "pointer",
        background: active ? "var(--accent)" : "var(--code-bg)",
        color: active ? "#fff" : "var(--text)",
    })

    const NAV_BTN = (disabled) => ({
        width: 32, height: 32, borderRadius: 6,
        border: "1px solid var(--border)",
        background: disabled ? "transparent" : "var(--code-bg)",
        color: disabled ? "var(--border)" : "var(--text-h)",
        cursor: disabled ? "default" : "pointer",
        fontSize: 16, display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0,
    })

    return (
        <div>
            {/* ── input area ── */}
            {!running && !done && (
                <>
                    <div style={{ fontSize: 13, color: "var(--text)", marginBottom: 8 }}>
                        Paste test sentences — one per line.{" "}
                        {parsed.length > 0 && (
                            <span style={{ color: "var(--accent)" }}>{parsed.length} sentence{parsed.length !== 1 ? "s" : ""} detected</span>
                        )}
                    </div>
                    <textarea
                        value={rawInput}
                        onChange={e => setRawInput(e.target.value)}
                        placeholder={"I had a slice of pavlova with fresh strawberries\nAte a meat pie and a can of L&P at lunch\nHad two Weet-Bix with trim milk this morning\n…"}
                        style={{
                            width: "100%", boxSizing: "border-box",
                            height: 220, padding: "12px 14px",
                            borderRadius: 8, border: "1px solid var(--border)",
                            background: "var(--code-bg)", color: "var(--text-h)",
                            fontFamily: "var(--mono)", fontSize: 13, lineHeight: "160%",
                            resize: "vertical", outline: "none",
                        }}
                    />
                    <div style={{ marginTop: 10, display: "flex", gap: 8, justifyContent: "flex-end" }}>
                        <button
                            onClick={handleRun}
                            disabled={!parsed.length}
                            style={{
                                padding: "9px 22px", borderRadius: 7, border: "none",
                                background: parsed.length ? "var(--accent)" : "var(--border)",
                                color: "#fff", cursor: parsed.length ? "pointer" : "default",
                                fontFamily: "var(--sans)", fontSize: 14, fontWeight: 500,
                            }}>
                            Run {parsed.length > 0 ? `${parsed.length} sentences` : ""}
                        </button>
                    </div>
                </>
            )}

            {/* ── progress bar while running ── */}
            {(running || done) && (
                <div style={{ marginBottom: 20 }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontSize: 12, color: "var(--text)" }}>
                            {running
                                ? `Processing ${completedCount + 1} of ${sentences.length}…`
                                : `Completed ${completedCount} of ${sentences.length}`}
                        </span>
                        <div style={{ display: "flex", gap: 8 }}>
                            {running && (
                                <button onClick={handleStop}
                                        style={{ padding: "4px 12px", borderRadius: 5, border: "1px solid var(--border)", background: "transparent", color: "var(--text)", fontSize: 12, cursor: "pointer" }}>
                                    Stop
                                </button>
                            )}
                            {done && (
                                <button onClick={() => { setDone(false); setSentences([]); setResults([]) }}
                                        style={{ padding: "4px 12px", borderRadius: 5, border: "1px solid var(--border)", background: "transparent", color: "var(--text)", fontSize: 12, cursor: "pointer" }}>
                                    ← New batch
                                </button>
                            )}
                        </div>
                    </div>
                    <div style={{ height: 4, borderRadius: 2, background: "var(--border)", overflow: "hidden" }}>
                        <div style={{
                            height: "100%", borderRadius: 2,
                            background: done ? "#1D9E75" : "var(--accent)",
                            width: `${progress * 100}%`,
                            transition: "width 0.25s ease",
                        }} />
                    </div>
                </div>
            )}

            {/* ── sentence navigator ── */}
            {sentences.length > 0 && (
                <>
                    {/* nav bar */}
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                        <button disabled={!canPrev} onClick={() => setCursor(c => c - 1)} style={NAV_BTN(!canPrev)}>‹</button>
                        <button disabled={!canNext} onClick={() => setCursor(c => c + 1)} style={NAV_BTN(!canNext)}>›</button>

                        {/* pill index strip */}
                        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", flex: 1 }}>
                            {sentences.map((_, i) => {
                                const done_i = results[i] !== null
                                const isErr  = results[i]?.source === "error"
                                return (
                                    <button key={i} onClick={() => setCursor(i)}
                                            title={sentences[i]}
                                            style={{
                                                width: 26, height: 26, borderRadius: 5,
                                                fontSize: 11, cursor: "pointer", fontFamily: "var(--mono)",
                                                background: i === cursor
                                                    ? "var(--accent)"
                                                    : isErr ? "#D85A3022"
                                                        : done_i ? "var(--code-bg)"
                                                            : "transparent",
                                                color: i === cursor ? "#fff" : done_i ? "var(--text-h)" : "var(--border)",
                                                border: i === cursor ? "none" : "1px solid var(--border)",
                                                fontWeight: i === cursor ? 600 : 400,
                                            }}>
                                        {i + 1}
                                    </button>
                                )
                            })}
                        </div>

                        <span style={{ fontSize: 11, color: "var(--text)", flexShrink: 0 }}>
                            {cursor + 1} / {sentences.length}
                        </span>
                    </div>

                    {/* current sentence label */}
                    <div style={{
                        fontSize: 15, padding: "10px 14px", borderRadius: 7,
                        border: "1px solid var(--border)", background: "var(--code-bg)",
                        color: "var(--text-h)", marginBottom: 16, lineHeight: "150%",
                    }}>
                        {sentences[cursor]}
                    </div>

                    {/* result or loading state */}
                    {results[cursor] === null ? (
                        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "24px 0", color: "var(--text)", fontSize: 13 }}>
                            <div style={{
                                width: 16, height: 16, borderRadius: "50%",
                                border: "2px solid var(--border)", borderTopColor: "var(--accent)",
                                animation: "spin 0.7s linear infinite", flexShrink: 0,
                            }} />
                            Waiting…
                        </div>
                    ) : (
                        <ResultView key={`${cursor}-${results[cursor].text}`} result={results[cursor]} />
                    )}
                </>
            )}
        </div>
    )
}

// ─── main App ─────────────────────────────────────────────────────────────────

export default function App() {
    const [mode, setMode]   = useState("single")  // "single" | "bulk"
    const [input, setInput] = useState("")
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState("")

    const handleExtract = async () => {
        const text = input.trim()
        if (!text || loading) return

        setLoading(true)
        setResult(null)
        setError("")
        try {
            const response = await fetch(`${API_URL}/extract`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            })
            if (!response.ok) throw new Error(`The API returned ${response.status}`)
            setResult(await response.json())
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unable to contact the API.")
        } finally {
            setLoading(false)
        }
    }

    const MODE_TAB = (active) => ({
        padding: "6px 18px", borderRadius: 6, fontSize: 13,
        border: "1px solid var(--border)", cursor: "pointer",
        background: active ? "var(--code-bg)" : "transparent",
        color: active ? "var(--text-h)" : "var(--text)",
        fontWeight: active ? 500 : 400,
    })

    return (
        <div style={{ padding: "40px 48px", textAlign: "left", maxWidth: 720, margin: "0 auto" }}>
            <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>

            <h1 style={{ marginBottom: 8, paddingBottom: 0 }}>Natural language input</h1>

            {/* mode switcher */}
            <div style={{ display: "flex", gap: 6, marginBottom: 28, marginTop: 16 }}>
                <button style={MODE_TAB(mode === "single")} onClick={() => setMode("single")}>Single</button>
                <button style={MODE_TAB(mode === "bulk")}   onClick={() => setMode("bulk")}>Bulk test</button>
            </div>

            {/* ── single mode ── */}
            {mode === "single" && (
                <>
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
                            disabled={!input.trim() || loading}
                            style={{
                                padding: "10px 20px", borderRadius: 7, border: "none",
                                background: !input.trim() || loading ? "var(--border)" : "var(--accent)",
                                color: "#fff", cursor: !input.trim() || loading ? "default" : "pointer",
                                fontFamily: "var(--sans)", fontSize: 15, fontWeight: 500,
                            }}>
                            Extract
                        </button>
                    </div>

                    {loading && (
                        <div style={{ display: "flex", justifyContent: "center", padding: "32px 0" }}>
                            <div style={{
                                width: 24, height: 24, borderRadius: "50%",
                                border: "2px solid var(--border)", borderTopColor: "var(--accent)",
                                animation: "spin 0.7s linear infinite",
                            }} />
                        </div>
                    )}

                    {error && (
                        <p style={{ color: "#b42318", marginTop: 0 }} role="alert">
                            Could not extract food items: {error}
                        </p>
                    )}

                    {result && (
                        <>
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                                <h2 style={{ margin: 0 }}>Result</h2>
                            </div>
                            <ResultView key={result.text} result={result} />
                        </>
                    )}
                </>
            )}

            {/* ── bulk mode ── */}
            {mode === "bulk" && <BulkTestPanel />}
        </div>
    )
}
