import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Upload } from "lucide-react";

interface AudioUploaderProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function AudioUploader({ onFileSelect, disabled }: AudioUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      setFileName(file.name);
      onFileSelect(file);
    } else {
      setFileName("");
    }
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  return (
    <div className="flex items-center gap-3">
      <Input
        ref={inputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="hidden"
      />
      <Button
        onClick={handleClick}
        disabled={disabled}
        variant="outline"
        className="gap-2"
      >
        <Upload className="h-4 w-4" />
        Upload Song
      </Button>
      {fileName && (
        <span className="text-sm text-muted-foreground truncate">{fileName}</span>
      )}
    </div>
  );
}
