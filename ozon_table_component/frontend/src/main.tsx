import React from 'react'
import ReactDOM from 'react-dom/client'
import { withStreamlitConnection } from 'streamlit-component-lib'
import App from './table/App'
import './style.css'

const Connected = withStreamlitConnection(App)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Connected />
  </React.StrictMode>,
)
