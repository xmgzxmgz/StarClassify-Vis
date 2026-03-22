import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AppHeader from "@/components/AppHeader";
import ToastViewport from "@/components/ToastViewport";
import Lab from "@/pages/Lab";
import Runs from "@/pages/Runs";
import { SettingsPage } from "@/pages/Settings";
import { ModeProvider } from "@/context/ModeContext";
import { ThemeProvider } from "@/context/ThemeContext";

export default function App() {
  return (
    <ThemeProvider>
      <ModeProvider>
        <Router>
          <div className="min-h-dvh bg-white dark:bg-slate-950 text-slate-900 dark:text-white transition-colors">
            <AppHeader />
            <Routes>
              <Route path="/" element={<Lab />} />
              <Route path="/runs" element={<Runs />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
            <ToastViewport />
          </div>
        </Router>
      </ModeProvider>
    </ThemeProvider>
  );
}
