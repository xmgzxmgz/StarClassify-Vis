import { create } from "zustand";

type Toast = {
  id: string;
  title: string;
  description?: string;
  createdAt: number;
};

type ToastState = {
  toasts: Toast[];
  push: (t: Omit<Toast, "id" | "createdAt">) => void;
  remove: (id: string) => void;
};

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (t) =>
    set((s) => {
      const id = crypto.randomUUID();
      const toast: Toast = { ...t, id, createdAt: Date.now() };
      return { toasts: [toast, ...s.toasts].slice(0, 3) };
    }),
  remove: (id) => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })),
}));
