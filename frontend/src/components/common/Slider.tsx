type Props = {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  label?: string;
};

export function Slider({ value, min = 0, max = 1, step = 0.01, onChange, label }: Props) {
  return (
    <label className="flex flex-col gap-1 text-xs text-slate-400">
      {label && <span>{label}</span>}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-sky-500"
      />
    </label>
  );
}
