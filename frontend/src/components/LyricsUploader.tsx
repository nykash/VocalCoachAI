import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload, X } from "lucide-react";

interface LyricsUploaderProps {
  onFileSelect: (file: File) => void;
  onClear?: () => void;
  disabled?: boolean;
  fileName?: string;
}

export default function LyricsUploader({ 
  onFileSelect, 
  onClear,
  disabled,
  fileName
}: LyricsUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Accept .lrc files or any text files
      if (file.name.endsWith(".lrc") || file.type === "text/plain" || file.type === "") {
        onFileSelect(file);
      } else {
        alert("Please select a .lrc or text file");
      }
    }
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleClear = () => {
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    onClear?.();
  };

  return (
    <div className="flex items-center gap-2">
      <input
        ref={inputRef}
        type="file"
        accept=".lrc,.txt,text/plain"
        onChange={handleFileChange}
        className="hidden"
      />
      <Button
        onClick={handleClick}
        disabled={disabled}
        variant="outline"
        size="sm"
        className="gap-2"
      >
        <Upload className="h-4 w-4" />
        Upload Lyrics
      </Button>
      {fileName && (
        <>
          <span className="text-sm text-muted-foreground truncate max-w-[200px]">
            {fileName}
          </span>
          <Button
            onClick={handleClear}
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0"
            title="Clear lyrics file"
          >
            <X className="h-3 w-3" />
          </Button>
        </>
      )}
    </div>
  );
}
