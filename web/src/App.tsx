import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import AppHeader from "@/components/AppHeader";
import ToastViewport from "@/components/ToastViewport";
import Lab from "@/pages/Lab";
import Runs from "@/pages/Runs";
import { ModeProvider } from "@/context/ModeContext";

export default function App() {
  return (
    <ModeProvider>
      <Router>
        <div className="min-h-dvh bg-[#0B1220] text-white">
          <AppHeader />
          <Routes>
            <Route path="/" element={<Lab />} />
            <Route path="/runs" element={<Runs />} />
          </Routes>
          <ToastViewport />
        </div>
      </Router>
    </ModeProvider>
  );
}
