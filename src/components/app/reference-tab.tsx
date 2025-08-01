"use client";

import Webcam from "react-webcam";
import { useEffect, useRef, useState } from "react";
import { generateDefectProfile } from "@/ai/flows/generate-defect-profile";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Bot, CheckCircle, Loader2 } from "lucide-react";

async function sendCommand(command: string) {
  await fetch("/api/motor", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command }),
  });
}

type ReferenceTabProps = {
  onModelTrained: (modelId: string) => void;
};

type Status = "idle" | "collecting" | "training" | "complete";

export function ReferenceTab({ onModelTrained }: ReferenceTabProps) {
  const webcamRef = useRef<Webcam>(null);
  const [showCamera, setShowCamera] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [trainedModelId, setTrainedModelId] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    let timer: NodeJS.Timeout | undefined;
    if (status === "training") {
      timer = setInterval(() => {
        setProgress((p) => (p >= 95 ? 95 : p + 1));
      }, 500);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [status]);

  const startProcess = async () => {
    setStatus("collecting");
    setProgress(0);
    setTrainedModelId(null);
    setShowCamera(true);
    await sendCommand("A1");
    await sendCommand("B1");
    const images: string[] = [];
    let count = 0;
    const interval = setInterval(() => {
      const img = webcamRef.current?.getScreenshot();
      if (img) images.push(img);
      count++;
      setProgress(Math.min(50, (count / 60) * 50));
    }, 1000);
    setTimeout(async () => {
      clearInterval(interval);
      await sendCommand("A0");
      await sendCommand("B0");
      setStatus("training");
      try {
        const result = await generateDefectProfile({ referenceImages: images });
        setTrainedModelId(result.modelId);
        onModelTrained(result.modelId);
        toast({
          title: "Model Trained Successfully",
          description: `New defect profile created with ID: ${result.modelId}`,
        });
        setProgress(100);
        setStatus("complete");
      } catch (error) {
        console.error("Error training model:", error);
        toast({
          variant: "destructive",
          title: "Training Failed",
          description: "An error occurred while training the AI model.",
        });
        setStatus("idle");
      } finally {
        setShowCamera(false);
      }
    }, 60000);
  };

  return (
    <Card className="max-w-3xl mx-auto shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl font-headline">
          <Bot /> Reference Mode: Train Defect Profile
        </CardTitle>
        <CardDescription>
          The system will collect images from the webcam for one minute while the
          motors rotate, then train a defect profile.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {showCamera && <Webcam audio={false} ref={webcamRef} className="w-full rounded-md" />}
        {trainedModelId && status === "complete" && (
          <div className="p-4 rounded-md bg-primary/10 border border-primary/20 flex items-center gap-3">
            <CheckCircle className="text-primary h-6 w-6" />
            <div>
              <p className="font-semibold text-primary">Training Complete!</p>
              <p className="text-sm text-primary/80 font-code">Model ID: {trainedModelId}</p>
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex flex-col gap-4">
        <Button onClick={startProcess} disabled={status === "collecting" || status === "training"} className="w-full">
          {status === "collecting" ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Collecting Data...
            </>
          ) : status === "training" ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Training Model...
            </>
          ) : (
            "Train Model"
          )}
        </Button>
        {(status === "collecting" || status === "training") && (
          <Progress value={progress} className="w-full" />
        )}
      </CardFooter>
    </Card>
  );
}
