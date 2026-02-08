import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { TuneMeModalProvider } from "@/contexts/TuneMeModalContext";
import { SingStyleModalProvider } from "@/contexts/SingStyleModalContext";
import { VocalRangeModalProvider } from "@/contexts/VocalRangeModalContext";
import Index from "./pages/Index";
import SingAlong from "./pages/SingAlong";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <TuneMeModalProvider>
        <SingStyleModalProvider>
          <VocalRangeModalProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/sing-along" element={<SingAlong />} />
                {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </BrowserRouter>
          </VocalRangeModalProvider>
        </SingStyleModalProvider>
      </TuneMeModalProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
