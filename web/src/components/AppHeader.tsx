import { Link, NavLink } from "react-router-dom";
import { FlaskConical, History } from "lucide-react";
import { cn } from "@/lib/utils";

export default function AppHeader() {
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
          <NavLink
            to="/"
            className={({ isActive }) =>
              cn(
                "rounded-md px-3 py-2 text-white/70 transition hover:bg-white/10 hover:text-white",
                isActive && "bg-white/10 text-white",
              )
            }
          >
            <span className="inline-flex items-center gap-2">
              <FlaskConical className="h-4 w-4" />
              实验台
            </span>
          </NavLink>
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
