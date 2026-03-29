import { Link, NavLink } from "react-router-dom";
import { FlaskConical, History, GraduationCap, Telescope, Microscope, Settings, Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";
import { useMode, UserMode } from "@/context/ModeContext";
import { useTheme } from "@/context/ThemeContext";

export default function AppHeader() {
  const { mode, setMode } = useMode();
  const { theme, toggleTheme } = useTheme();

  const modes: { id: UserMode; label: string; icon: React.ElementType }[] = [
    { id: "researcher", label: "科研人员", icon: Microscope },
    { id: "educator", label: "天文师生", icon: GraduationCap },
    { id: "public", label: "科普人员", icon: Telescope },
  ];

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-950/80 backdrop-blur transition-colors">
      <div className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-6">
        <Link
          to="/"
          className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-white"
        >
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 ring-1 ring-slate-200 dark:ring-slate-700">
            <FlaskConical className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
          </span>
          恒星分类可视化实验台
        </Link>

        <nav className="flex items-center gap-2 text-sm">
          <div className="mr-4 flex items-center rounded-lg bg-slate-100 dark:bg-slate-900 p-1 ring-1 ring-slate-200 dark:ring-slate-800">
            {modes.map((m) => (
              <button
                key={m.id}
                onClick={() => setMode(m.id)}
                className={cn(
                  "flex items-center gap-2 rounded-md px-3 py-1.5 transition-all",
                  mode === m.id
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white"
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
                "rounded-md px-3 py-2 text-slate-600 dark:text-slate-400 transition hover:bg-slate-100 dark:hover:bg-slate-900 hover:text-slate-900 dark:hover:text-white",
                isActive && "bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-white",
              )
            }
          >
            <span className="inline-flex items-center gap-2">
              <History className="h-4 w-4" />
              结果记录
            </span>
          </NavLink>

          {/* 主题切换按钮 */}
          <button
            onClick={toggleTheme}
            className="rounded-md px-3 py-2 text-slate-600 dark:text-slate-400 transition hover:bg-slate-100 dark:hover:bg-slate-900 hover:text-slate-900 dark:hover:text-white"
            aria-label="切换主题"
            title={theme === 'light' ? '切换到深色模式' : '切换到浅色模式'}
          >
            {theme === 'light' ? (
              <Moon className="h-4 w-4" />
            ) : (
              <Sun className="h-4 w-4" />
            )}
          </button>

          {/* 设置链接 */}
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              cn(
                "rounded-md px-3 py-2 text-slate-600 dark:text-slate-400 transition hover:bg-slate-100 dark:hover:bg-slate-900 hover:text-slate-900 dark:hover:text-white",
                isActive && "bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-white",
              )
            }
          >
            <Settings className="h-4 w-4" />
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
