import { Mic, MicOff } from "lucide-react";
import { Button } from "@/components/ui/button";

interface MicButtonProps {
  isListening: boolean;
  onToggle: () => void;
}

const MicButton = ({ isListening, onToggle }: MicButtonProps) => {
  return (
    <div className="relative">
      {/* Animated rings when listening */}
      {isListening && (
        <>
          <span className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />
          <span className="absolute -inset-3 rounded-full pulse-ring" />
        </>
      )}

      <Button
        onClick={onToggle}
        size="lg"
        className={`
          relative z-10 h-20 w-20 rounded-full text-primary-foreground
          transition-all duration-300 shadow-lg
          ${isListening
            ? "bg-primary hover:bg-primary/90 scale-110 glow-primary"
            : "bg-muted-foreground/20 hover:bg-primary/80 text-foreground hover:text-primary-foreground"
          }
        `}
      >
        {isListening ? (
          <MicOff className="h-8 w-8" />
        ) : (
          <Mic className="h-8 w-8" />
        )}
      </Button>
    </div>
  );
};

export default MicButton;
