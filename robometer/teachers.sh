# The RoboMeter teachers of the campaign. Sourced (by robometer/serve.sh and value/fit_v.sh), never executed.
#
#   teacher <name>   sets TEACHER_TASK, TEACHER_LORA, TEACHER_EXP, TEACHER_TEXT for one teacher; returns 1 if unknown
#   teacher_model    prints the --model path of the selected teacher
#
# One teacher = one long-lived robometer_server.py process on its own spool directory $SPOOL_ROOT/<name>.
#
#   name               LoRA (Hugging Face repo name)            what it is used for
#   threading_d0       robometer-4b-lora-threading-d0-seg       labels threading pv4 and pv4_seg (the two score tables are
#                                                               identical: the same LoRA scored the same rows twice);
#                                                               online teacher of every threading TD+distill cell
#   square_d0          robometer-4b-lora-square-d0-seg          labels square_d0 pv4; online teacher of square_d0
#   hammer_cleanup_d1  robometer-4b-lora-hammer-cleanup-seg     labels hammer pv4; online teacher of hammer
#   square_d1          robometer-4b-lora-square-d1              the NON-segmented LoRA: labels square_d1 pv4 (the V of the
#                                                               DICE-RL square_d1 cells) and is the online teacher of
#                                                               their TD+distill variant
#   square_d1_seg      robometer-4b-lora-square-d1-seg-c095     the segmented LoRA: labels square_d1 pv4_seg (the V of
#                                                               the DSRL square_d1 cells). Offline rescoring only: no
#                                                               cell of the campaign distils from it online
#
# The first four are the online teachers, one per task, on $SPOOL_ROOT/<task> (the same table as configs/tasks.sh;
# scripts/robometer_teacher.sh starts and checks them by task). A TD+distill cell must distil from the teacher that
# labelled the V it starts from: that holds for every such cell of the campaign (DICE-RL on all four tasks, DSRL on
# square_d0 and hammer_cleanup_d1). The DSRL square_d1 cells start from pv4_seg and never distil; a TD+distill
# variant of them would need square_d1_seg as its spool, not the task's default.
# TEACHER_TEXT is the sentence the LoRA was fine-tuned with (the task dataset's meta/tasks.parquet entry). It is not the
# env's TaskSpec.instruction for the square and hammer tasks; a wrong sentence still returns plausible scores.
# TEACHER_EXP is training.exp_name of the fine-tune (robometer/finetune/train_lora.sh), i.e. the directory a local
# fine-tune writes; the non-segmented square_d1 LoRA was not fine-tuned with this pipeline.

teacher() {
  case "${1:-}" in
    threading_d0)
      TEACHER_TASK=threading_d0 TEACHER_LORA=robometer-4b-lora-threading-d0-seg TEACHER_EXP=threading_d0_seg_lora
      TEACHER_TEXT="Pick up the needle and thread it through the hole in the tripod." ;;
    square_d0)
      TEACHER_TASK=square_d0 TEACHER_LORA=robometer-4b-lora-square-d0-seg TEACHER_EXP=square_d0_seg_lora
      TEACHER_TEXT="Pick up the square nut and place it on the peg." ;;
    hammer_cleanup_d1)
      TEACHER_TASK=hammer_cleanup_d1 TEACHER_LORA=robometer-4b-lora-hammer-cleanup-seg TEACHER_EXP=hammer_cleanup_seg_lora
      TEACHER_TEXT="Open the drawer, place the hammer inside, and close the drawer." ;;
    square_d1)
      TEACHER_TASK=square_d1 TEACHER_LORA=robometer-4b-lora-square-d1 TEACHER_EXP=
      TEACHER_TEXT="Pick up the square nut and place it on the peg." ;;
    square_d1_seg)
      TEACHER_TASK=square_d1 TEACHER_LORA=robometer-4b-lora-square-d1-seg-c095 TEACHER_EXP=square_d1_seg_c095_lora
      TEACHER_TEXT="Pick up the square nut and place it on the peg." ;;
    *)
      echo "unknown teacher '${1:-}': one of threading_d0 square_d0 hammer_cleanup_d1 square_d1 square_d1_seg" >&2
      return 1 ;;
  esac
}

# --model of the selected teacher: $CKPT_ROOT/robometer/<LoRA>/, or the ckpt-latest-* directory inside it when the
# directory is the output of a local fine-tune (train.py writes <out>/<exp>/ckpt-latest-avg-.../). The trailing slash is
# how the reference servers were started.
teacher_model() {
  local dir="$CKPT_ROOT/robometer/$TEACHER_LORA" latest
  latest=$(ls -d "$dir"/ckpt-latest-* "$dir/$TEACHER_EXP"/ckpt-latest-* 2>/dev/null | head -n 1 || true)
  echo "${latest:-$dir}/"
}
