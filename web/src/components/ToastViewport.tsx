import { X } from "lucide-react";
import { useEffect } from "react";
import { cn } from "@/lib/utils";
import { useToastStore } from "@/hooks/useToast";

export default function ToastViewport() {
  const toasts = useToastStore((s) => s.toasts);
  const remove = useToastStore((s) => s.remove);

  useEffect(() => {
    if (toasts.length === 0) return;
    const t = setInterval(() => {
      const now = Date.now();
      for (const toast of toasts) {
        if (now - toast.createdAt > 3500) remove(toast.id);
      }
    }, 300);
    return () => clearInterval(t);
  }, [toasts, remove]);

  return (
    <div className="fixed right-4 top-16 z-50 flex w-[360px] flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={cn(
            "rounded-xl border border-white/10 bg-[#111A2E] p-3 shadow-lg",
          )}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="text-sm font-semibold text-white">{t.title}</div>
              {t.description ? (
                <div className="mt-0.5 text-sm text-white/60">
                  {t.description}
                </div>
              ) : null}
            </div>
            <button
              onClick={() => remove(t.id)}
              className="rounded-md p-1 text-white/60 transition hover:bg-white/10 hover:text-white"
              type="button"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
