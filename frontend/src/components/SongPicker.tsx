import { useState, useEffect, useMemo } from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { fetchSongList } from "@/lib/songsApi";
import { cn } from "@/lib/utils";

interface SongPickerProps {
  onSelect: (filename: string) => void;
  selectedFilename: string | null;
  disabled?: boolean;
  placeholder?: string;
}

export default function SongPicker({
  onSelect,
  selectedFilename,
  disabled,
  placeholder = "Pick a song…",
}: SongPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [songs, setSongs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open && songs.length === 0) {
      setLoading(true);
      setError(null);
      fetchSongList()
        .then(setSongs)
        .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
        .finally(() => setLoading(false));
    }
  }, [open, songs.length]);

  const filtered = useMemo(() => {
    if (!search.trim()) return songs;
    const q = search.trim().toLowerCase();
    return songs.filter((name) => name.toLowerCase().includes(q));
  }, [songs, search]);

  const displayLabel = selectedFilename
    ? selectedFilename.replace(/\.(mp3|m4a|wav|flac|ogg)$/i, "")
    : null;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          className="w-full justify-between font-normal"
        >
          <span className={cn("truncate", !displayLabel && "text-muted-foreground")}>
            {displayLabel ?? placeholder}
          </span>
          <ChevronDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0" align="start">
        <div className="p-2 border-b">
          <Input
            placeholder="Search songs…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-9"
          />
        </div>
        <ScrollArea className="h-[280px]">
          {loading && (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          )}
          {error && (
            <div className="px-3 py-4 text-sm text-destructive">{error}</div>
          )}
          {!loading && !error && filtered.length === 0 && (
            <div className="px-3 py-4 text-sm text-muted-foreground">
              No songs found.
            </div>
          )}
          {!loading && filtered.length > 0 && (
            <ul className="p-1">
              {filtered.map((filename) => (
                <li key={filename}>
                  <button
                    type="button"
                    className={cn(
                      "w-full rounded-md px-3 py-2 text-left text-sm truncate",
                      "hover:bg-accent hover:text-accent-foreground",
                      selectedFilename === filename && "bg-accent text-accent-foreground"
                    )}
                    onClick={() => {
                      onSelect(filename);
                      setOpen(false);
                      setSearch("");
                    }}
                  >
                    {filename.replace(/\.(mp3|m4a|wav|flac|ogg)$/i, "")}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </ScrollArea>
      </PopoverContent>
    </Popover>
  );
}
