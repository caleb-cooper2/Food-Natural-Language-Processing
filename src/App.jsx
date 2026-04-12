import './App.css'
import {useState} from "react";

function App() {
    const [input, setInput] = useState('')
    const [output, setOutput] = useState('')

    const handleExtract = async () => {
        fetch('http://localhost:8000/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input })
        })
            .then(response => response.json())
            .then(json => setOutput(JSON.stringify(json.data)))
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

            <h2 style={{marginTop: 20}}>Output:</h2>
            <textarea rows={20} readOnly value={output}/>
        </>
    )
}

export default App
