import './App.css'
import {useState} from "react";

function App() {
    const [input, setInput] = useState('')
    const handleExtract = async () => {
        fetch('http://localhost:8000/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input })
        })
            .then(response => response.json())
            .then(json => console.info(json))
            .catch(error => console.error('Error fetching data:', error));
    }

    return (
        <>
            <h1>Natural language input</h1>
            <textarea rows={2}
                      placeholder="e.g. 2 scrambled eggs and a piece of toast..."
                      value={input}
                      onChange={e => setInput(e.target.value)} />
            <button onClick={handleExtract}>Extract</button>
        </>
    )
}

export default App
