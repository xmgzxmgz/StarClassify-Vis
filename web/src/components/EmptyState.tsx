import { cn } from "@/lib/utils";

export default function EmptyState(props: {
  title: string;
  description: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-slate-200 dark:border-dashed dark:border-white/15 bg-slate-50 dark:bg-white/5 p-6",
        props.className,
      )}
    >
      <div className="text-sm font-semibold text-slate-900 dark:text-white">{props.title}</div>
      <div className="mt-1 text-sm text-slate-600 dark:text-white/60">{props.description}</div>
    </div>
  );
}
