import { createContext, useContext, useState, ReactNode } from "react";

export type UserMode = "researcher" | "educator" | "public";

interface ModeContextType {
  mode: UserMode;
  setMode: (mode: UserMode) => void;
}

const ModeContext = createContext<ModeContextType | undefined>(undefined);

export function ModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<UserMode>("researcher");

  return (
    <ModeContext.Provider value={{ mode, setMode }}>
      {children}
    </ModeContext.Provider>
  );
}

export function useMode() {
  const context = useContext(ModeContext);
  if (context === undefined) {
    throw new Error("useMode must be used within a ModeProvider");
  }
  return context;
}
