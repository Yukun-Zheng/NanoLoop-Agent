import { Slot } from "@radix-ui/react-slot";
import { clsx } from "clsx";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  asChild?: boolean;
  tone?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
};

export function Button({
  asChild,
  tone = "secondary",
  size = "md",
  className,
  ...props
}: ButtonProps) {
  const Component = asChild ? Slot : "button";
  return (
    <Component
      className={clsx("button", `button-${tone}`, `button-${size}`, className)}
      {...props}
    />
  );
}
