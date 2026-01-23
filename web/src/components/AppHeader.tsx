import { Link, NavLink } from "react-router-dom";
import { FlaskConical, History, GraduationCap, Telescope, Microscope } from "lucide-react";
import { cn } from "@/lib/utils";
import { useMode, UserMode } from "@/context/ModeContext";

export default function AppHeader() {
  const { mode, setMode } = useMode();

  const modes: { id: UserMode; label: string; icon: React.ElementType }[] = [
    { id: "researcher", label: "科研人员", icon: Microscope },
    { id: "educator", label: "天文师生", icon: GraduationCap },
    { id: "public", label: "科普人员", icon: Telescope },
  ];

  return (
    <header className="sticky top-0 z-20 border-b border-white/10 bg-[#0B1220]/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-6">
        <Link
          to="/"
          className="flex items-center gap-2 text-sm font-semibold text-white"
        >
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-white/10 ring-1 ring-white/10">
            <FlaskConical className="h-4 w-4" />
          </span>
          星体分类可视化实验台
        </Link>

        <nav className="flex items-center gap-2 text-sm">
          <div className="mr-4 flex items-center rounded-lg bg-white/5 p-1 ring-1 ring-white/10">
            {modes.map((m) => (
              <button
                key={m.id}
                onClick={() => setMode(m.id)}
                className={cn(
                  "flex items-center gap-2 rounded-md px-3 py-1.5 transition-all",
                  mode === m.id
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-white/60 hover:bg-white/5 hover:text-white"
                )}
              >
                <m.icon className="h-4 w-4" />
                {m.label}
              </button>
            ))}
          </div>

          <NavLink
            to="/runs"
            className={({ isActive }) =>
              cn(
                "rounded-md px-3 py-2 text-white/70 transition hover:bg-white/10 hover:text-white",
                isActive && "bg-white/10 text-white",
              )
            }
          >
            <span className="inline-flex items-center gap-2">
              <History className="h-4 w-4" />
              结果记录
            </span>
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
