import { cn } from "@/lib/utils";

export default function Card(props: {
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn(
        "rounded-xl border border-slate-200 dark:border-white/10 bg-white dark:bg-[#111A2E] p-4 shadow-sm",
        props.className,
      )}
    >
      {props.title ? (
        <h2 className="mb-3 text-sm font-semibold text-slate-900 dark:text-white">{props.title}</h2>
      ) : null}
      {props.children}
    </section>
  );
}
