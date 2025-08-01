"use client";

import Webcam from "react-webcam";
import { useEffect, useRef, useState, type MouseEvent } from "react";
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
  const overlayRef = useRef<HTMLDivElement>(null);
  const [showCamera, setShowCamera] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [trainedModelId, setTrainedModelId] = useState<string | null>(null);
  const [trainingDuration, setTrainingDuration] = useState(60);
  const [selecting, setSelecting] = useState(false);
  const [cropRect, setCropRect] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
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

  const beginSelect = (e: MouseEvent<HTMLDivElement>) => {
    if (status !== "idle") return;
    const rect = overlayRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setCropRect({ x, y, width: 0, height: 0 });
    setSelecting(true);
  };

  const updateSelect = (e: MouseEvent<HTMLDivElement>) => {
    if (!selecting || !cropRect) return;
    const rect = overlayRef.current!.getBoundingClientRect();
    const width = e.clientX - rect.left - cropRect.x;
    const height = e.clientY - rect.top - cropRect.y;
    setCropRect({ ...cropRect, width, height });
  };

  const endSelect = () => {
    setSelecting(false);
  };

  async function cropImage(dataUri: string): Promise<string> {
    if (!cropRect) return dataUri;
    const img = new Image();
    img.src = dataUri;
    await new Promise<void>(res => { img.onload = () => res(); });
    const canvas = document.createElement("canvas");
    const videoW = img.width;
    const videoH = img.height;
    const overlay = overlayRef.current?.getBoundingClientRect();
    const scaleX = overlay ? videoW / overlay.width : 1;
    const scaleY = overlay ? videoH / overlay.height : 1;
    const x = Math.min(cropRect.x, cropRect.x + cropRect.width) * scaleX;
    const y = Math.min(cropRect.y, cropRect.y + cropRect.height) * scaleY;
    const w = Math.abs(cropRect.width) * scaleX;
    const h = Math.abs(cropRect.height) * scaleY;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, x, y, w, h, 0, 0, w, h);
    return canvas.toDataURL("image/png");
  }

  const startProcess = async () => {
    setStatus("collecting");
    setProgress(0);
    setTrainedModelId(null);
    setShowCamera(true);
    await sendCommand("A1");
    await sendCommand("B1");
    const images: string[] = [];
    let count = 0;
    const interval = setInterval(async () => {
      const img = webcamRef.current?.getScreenshot();
      if (img) images.push(await cropImage(img));
      count++;
      setProgress(Math.min(50, (count / trainingDuration) * 50));
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
    }, trainingDuration * 1000);
  };

  return (
    <Card className="max-w-3xl mx-auto shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl font-headline">
          <Bot /> Reference Mode: Train Defect Profile
        </CardTitle>
        <CardDescription>
          Select an area of interest on the camera preview and specify how long
          to capture images. The system will collect frames from only that
          region while the motors rotate, then train a defect profile.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {showCamera && (
          <div className="relative">
            <Webcam audio={false} ref={webcamRef} className="w-full rounded-md" />
            <div
              ref={overlayRef}
              className="absolute inset-0 cursor-crosshair"
              onMouseDown={beginSelect}
              onMouseMove={updateSelect}
              onMouseUp={endSelect}
              onMouseLeave={endSelect}
            >
              {cropRect && (
                <div
                  className="absolute border border-red-500"
                  style={{
                    left: Math.min(cropRect.x, cropRect.x + cropRect.width),
                    top: Math.min(cropRect.y, cropRect.y + cropRect.height),
                    width: Math.abs(cropRect.width),
                    height: Math.abs(cropRect.height),
                  }}
                />
              )}
            </div>
          </div>
        )}
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
        <div className="flex items-center gap-2 w-full">
          <label className="text-sm whitespace-nowrap" htmlFor="trainTime">Training Time (s)</label>
          <input
            id="trainTime"
            type="number"
            min={1}
            className="border rounded px-2 py-1 flex-grow"
            value={trainingDuration}
            onChange={e => setTrainingDuration(Number(e.target.value))}
            disabled={status !== "idle"}
          />
        </div>
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
