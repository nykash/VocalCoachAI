import { createContext, useCallback, useContext, useRef, useState } from "react";
import SingStyleModal, { type SingStyleResult } from "@/components/SingStyleModal";

export type { SingStyleResult };

type OnResult = (result: SingStyleResult) => void;

interface SingStyleModalContextValue {
  /** Opens the Sing Style modal; returns a Promise that resolves with min/avg cent error and style tags when closed. */
  showSingStyleModal: (onResult?: OnResult) => Promise<SingStyleResult>;
}

export const SingStyleModalContext = createContext<SingStyleModalContextValue | null>(null);

export function SingStyleModalProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const onResultRef = useRef<OnResult | null>(null);
  const resolveRef = useRef<((result: SingStyleResult) => void) | null>(null);

  const showSingStyleModal = useCallback((onResult?: OnResult) => {
    onResultRef.current = onResult ?? null;
    setOpen(true);
    return new Promise<SingStyleResult>((resolve) => {
      resolveRef.current = resolve;
    });
  }, []);

  const handleClose = useCallback((result: SingStyleResult) => {
    onResultRef.current?.(result);
    onResultRef.current = null;
    resolveRef.current?.(result);
    resolveRef.current = null;
  }, []);

  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next);
  }, []);

  return (
    <SingStyleModalContext.Provider value={{ showSingStyleModal }}>
      {children}
      <SingStyleModal open={open} onOpenChange={handleOpenChange} onClose={handleClose} />
    </SingStyleModalContext.Provider>
  );
}

export function useSingStyleModal(): SingStyleModalContextValue {
  const ctx = useContext(SingStyleModalContext);
  if (!ctx) throw new Error("useSingStyleModal must be used within SingStyleModalProvider");
  return ctx;
}
