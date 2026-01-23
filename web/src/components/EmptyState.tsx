import { cn } from "@/lib/utils";

export default function EmptyState(props: {
  title: string;
  description: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-dashed border-white/15 bg-white/5 p-6",
        props.className,
      )}
    >
      <div className="text-sm font-semibold text-white">{props.title}</div>
      <div className="mt-1 text-sm text-white/60">{props.description}</div>
    </div>
  );
}
