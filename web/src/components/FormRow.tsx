import { cn } from "@/lib/utils";

export default function FormRow(props: {
  label: string;
  hint?: string;
  error?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1.5", props.className)}>
      <div className="flex items-baseline justify-between gap-3">
        <label className="text-xs font-medium text-white/80">
          {props.label}
        </label>
        {props.hint ? (
          <span className="text-xs text-white/40">{props.hint}</span>
        ) : null}
      </div>
      {props.children}
      {props.error ? (
        <div className="text-xs text-red-400">{props.error}</div>
      ) : null}
    </div>
  );
}
