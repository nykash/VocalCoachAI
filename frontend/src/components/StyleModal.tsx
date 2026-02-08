import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { fetchVaeTags, type VaeTagResult } from "@/lib/analysisApi";

interface StyleModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  getRecordedBlob: () => Blob | null;
}

export default function StyleModal({
  open,
  onOpenChange,
  getRecordedBlob,
}: StyleModalProps) {
  const [result, setResult] = useState<VaeTagResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch VAE tags when the modal opens
  useEffect(() => {
    if (!open) return;
    setResult(null);
    setError(null);

    const blob = getRecordedBlob();
    if (!blob || blob.size === 0) {
      setError("No recorded audio available. Make sure your mic is on and you've been singing.");
      return;
    }

    setLoading(true);
    fetchVaeTags(blob)
      .then((r) => setResult(r))
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Style analysis failed")
      )
      .finally(() => setLoading(false));
  }, [open, getRecordedBlob]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Singing Style Analysis</DialogTitle>
          <DialogDescription>
            Vocal style analysis from your recorded audio
          </DialogDescription>
        </DialogHeader>

        {loading && (
          <div className="flex items-center gap-3 py-6 justify-center text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="text-sm">Analyzing your singing style...</span>
          </div>
        )}

        {error && (
          <p className="text-sm text-destructive py-4">{error}</p>
        )}

        {result && !loading && (
          <div className="space-y-4">
            {/* Closest artist match */}
            {result.top_artist && (
              <div>
                <p className="text-xs text-muted-foreground">Closest artist match</p>
                <p className="text-lg font-semibold">{result.top_artist}</p>
              </div>
            )}

            {/* Top 3 artists */}
            {result.top_3_artists.length > 1 && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">
                  Similar artists
                </p>
                <div className="flex flex-wrap gap-2">
                  {result.top_3_artists.map((artist) => (
                    <span
                      key={artist}
                      className="rounded-full border px-3 py-1 text-sm"
                    >
                      {artist}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Style tags */}
            {result.top_3_attributes.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">
                  Style tags
                </p>
                <div className="flex flex-wrap gap-2">
                  {result.top_3_attributes.map(({ tag, confidence }) => (
                    <span
                      key={tag}
                      className="rounded-full bg-primary/15 px-3 py-1 text-sm font-medium text-primary"
                    >
                      {tag} ({(confidence * 100).toFixed(0)}%)
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
