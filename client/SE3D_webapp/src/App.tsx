import {graphData} from "./Data/graphs"
import {useState} from "react"
import SpectrumGraphItem from "./Components/spectrumGraphItem"

function App() {

  const [graphs] = useState(graphData)
  
  return (
    <main className="min-h-screen" style={{
      backgroundColor: "#36465d"
    }}>
      <div className = "space-y-2">
        { graphs.map((graph) => (
            <SpectrumGraphItem 
              spectrumGraph={graph}
            />
          ))
        }
      </div>
    </main>
  )
}

export default App
