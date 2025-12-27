import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// import App from './App.jsx'  // v1 SNN system (port 8000)
import GenesisApp from './GenesisApp.jsx'  // Genesis FEP system (port 8002)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <GenesisApp />
  </StrictMode>,
)
