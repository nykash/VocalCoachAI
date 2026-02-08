import { createContext, useCallback, useContext, useRef, useState } from "react";
import VocalRangeModal, { type VocalRangeResult } from "@/components/VocalRangeModal";

export type { VocalRangeResult };

type OnResult = (result: VocalRangeResult) => void;

interface VocalRangeModalContextValue {
  /** Opens the Vocal Range modal; returns a Promise that resolves with lowNote and highNote when closed. */
  showVocalRangeModal: (onResult?: OnResult) => Promise<VocalRangeResult>;
}

export const VocalRangeModalContext = createContext<VocalRangeModalContextValue | null>(null);

export function VocalRangeModalProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const onResultRef = useRef<OnResult | null>(null);
  const resolveRef = useRef<((result: VocalRangeResult) => void) | null>(null);

  const showVocalRangeModal = useCallback((onResult?: OnResult) => {
    onResultRef.current = onResult ?? null;
    setOpen(true);
    return new Promise<VocalRangeResult>((resolve) => {
      resolveRef.current = resolve;
    });
  }, []);

  const handleClose = useCallback((result: VocalRangeResult) => {
    onResultRef.current?.(result);
    onResultRef.current = null;
    resolveRef.current?.(result);
    resolveRef.current = null;
  }, []);

  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next);
  }, []);

  return (
    <VocalRangeModalContext.Provider value={{ showVocalRangeModal }}>
      {children}
      <VocalRangeModal open={open} onOpenChange={handleOpenChange} onClose={handleClose} />
    </VocalRangeModalContext.Provider>
  );
}

export function useVocalRangeModal(): VocalRangeModalContextValue {
  const ctx = useContext(VocalRangeModalContext);
  if (!ctx) throw new Error("useVocalRangeModal must be used within VocalRangeModalProvider");
  return ctx;
}
