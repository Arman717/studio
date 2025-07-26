"use client";

import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Loader2 } from "lucide-react";

interface TrainingDialogProps {
  open: boolean;
}

export function TrainingDialog({ open }: TrainingDialogProps) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!open) {
      setProgress(0);
      return;
    }

    const id = setInterval(() => {
      setProgress((prev) => (prev >= 90 ? 10 : prev + 10));
    }, 500);
    return () => clearInterval(id);
  }, [open]);

  return (
    <Dialog open={open}>
      <DialogContent className="flex flex-col items-center gap-4">
        <DialogHeader className="text-center">
          <DialogTitle>Training Model</DialogTitle>
          <DialogDescription>
            Hang tight! We&apos;re training your AI model.
          </DialogDescription>
        </DialogHeader>
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <Progress value={progress} className="w-full" />
      </DialogContent>
    </Dialog>
  );
}
