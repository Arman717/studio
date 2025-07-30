"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";

async function sendCommand(command: string) {
  await fetch("/api/motor", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command }),
  });
}

function MotorControl({ label, prefix }: { label: string; prefix: string }) {
  const [speed, setSpeed] = useState(200);

  const updateSpeed = async (val: number[]) => {
    const s = val[0];
    setSpeed(s);
    await sendCommand(`${prefix === "A" ? "SA" : "SB"}${s}`);
  };

  const sendDir = async (dir: string) => {
    await sendCommand(`${prefix}${dir}`);
  };

  return (
    <Card className="shadow-lg">
      <CardHeader>
        <CardTitle className="text-lg font-headline">{label}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Button onClick={() => sendDir("1")}>Forward</Button>
          <Button onClick={() => sendDir("2")}>Backward</Button>
          <Button onClick={() => sendDir("0")}>Stop</Button>
        </div>
        <div>
          <p className="text-sm mb-2">Speed: {speed}</p>
          <Slider min={0} max={255} value={[speed]} onValueChange={updateSpeed} />
        </div>
      </CardContent>
    </Card>
  );
}

export function ControlTab() {
  return (
    <div className="grid md:grid-cols-2 gap-8">
      <MotorControl label="Motor A" prefix="A" />
      <MotorControl label="Motor B" prefix="B" />
    </div>
  );
}

