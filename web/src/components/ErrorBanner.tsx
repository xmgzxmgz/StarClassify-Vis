export default function ErrorBanner(props: {
  title?: string;
  message: string;
}) {
  return (
    <div className="rounded-lg border border-red-300 dark:border-red-500/30 bg-red-50 dark:bg-red-500/10 p-3">
      {props.title ? (
        <div className="text-sm font-semibold text-red-800 dark:text-red-200">{props.title}</div>
      ) : null}
      <div className="text-sm text-red-700 dark:text-red-200/80">{props.message}</div>
    </div>
  );
}
