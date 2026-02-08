import { createContext, useCallback, useContext, useRef, useState } from "react";
import TuneMeModal, { type TuneMeResult } from "@/components/TuneMeModal";

export type { TuneMeResult };

type OnResult = (result: TuneMeResult) => void;

interface TuneMeModalContextValue {
  /** Opens the Tune Me modal and returns a Promise that resolves with min/avg error (cents) when the modal is closed. */
  showTuneMeModal: (onResult?: OnResult) => Promise<TuneMeResult>;
}

export const TuneMeModalContext = createContext<TuneMeModalContextValue | null>(null);

export function TuneMeModalProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const onResultRef = useRef<OnResult | null>(null);
  const resolveRef = useRef<((result: TuneMeResult) => void) | null>(null);

  const showTuneMeModal = useCallback((onResult?: OnResult) => {
    onResultRef.current = onResult ?? null;
    setOpen(true);
    return new Promise<TuneMeResult>((resolve) => {
      resolveRef.current = resolve;
    });
  }, []);

  const handleClose = useCallback((result: TuneMeResult) => {
    onResultRef.current?.(result);
    onResultRef.current = null;
    resolveRef.current?.(result);
    resolveRef.current = null;
    // Modal will call onOpenChange(false) after this, which sets open to false
  }, []);

  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next);
  }, []);

  return (
    <TuneMeModalContext.Provider value={{ showTuneMeModal }}>
      {children}
      <TuneMeModal
        open={open}
        onOpenChange={handleOpenChange}
        onClose={handleClose}
      />
    </TuneMeModalContext.Provider>
  );
}

export function useTuneMeModal(): TuneMeModalContextValue {
  const ctx = useContext(TuneMeModalContext);
  if (!ctx) throw new Error("useTuneMeModal must be used within TuneMeModalProvider");
  return ctx;
}
