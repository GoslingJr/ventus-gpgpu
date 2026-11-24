package play.cache

import chisel3._
import chiseltest._
import org.scalatest.freespec.AnyFreeSpec
import top.parameters._
import L1Cache.AtomicUnit.{AtomicUnit, inFlightWrite}
import L1Cache.MyConfig

class AtomicUnitTest extends AnyFreeSpec with ChiselScalatestTester {
  private val config = (new MyConfig).toInstance

  "retain both write tags when a request and a response fire together" in {
    test(new inFlightWrite()(config)) { dut =>
      val io = dut.io
      io.remvEntry.valid.poke(false.B)
      io.setEntryisSC.poke(false.B)
      io.setEntry.valid.poke(true.B)
      io.setEntry.bits.a_addr.poke(0x100.U)
      val first = io.SetIdx.peek().litValue
      dut.clock.step()

      io.setEntry.bits.a_addr.poke(0x200.U)
      val second = io.SetIdx.peek().litValue
      io.remvEntry.valid.poke(true.B)
      io.remvEntry.bits.poke(first.U)
      dut.clock.step()

      io.remvEntry.valid.poke(false.B)
      io.setEntry.bits.a_addr.poke(0x300.U)
      assert(io.SetIdx.peek().litValue != second)
      dut.clock.step()
      io.setEntry.valid.poke(false.B)
      io.checkEntry.a_addr.poke(0x200.U)
      io.conflict.expect(true.B)
      io.checkEntry.a_addr.poke(0x300.U)
      io.conflict.expect(true.B)
      io.checkEntry.a_addr.poke(0x100.U)
      io.conflict.expect(false.B)
    }
  }

  "return the old word once, after the masked write has been acknowledged" in {
    test(new AtomicUnit(l2cache_params)(config)) { dut =>
      val request = dut.io.L12ATUmemReq
      val toL2 = dut.io.ATU2L2memReq
      val fromL2 = dut.io.L22ATUmemRsp
      val result = dut.io.ATU2L1memRsp
      val word = 3
      val mask = BigInt(15) << (4 * word)
      val oldLine = (BigInt(19) << (32 * word)) | 123
      request.valid.poke(false.B)
      fromL2.valid.poke(false.B)
      toL2.ready.poke(false.B)
      result.ready.poke(false.B)
      dut.clock.step()

      request.bits.opcode.poke(2.U) // TileLink ArithmeticData
      request.bits.param.poke(4.U)  // ADD
      request.bits.size.poke(2.U)
      request.bits.source.poke(1.U)
      request.bits.address.poke(0x100.U)
      request.bits.mask.poke(mask.U)
      request.bits.data.poke((BigInt(7) << (32 * word)).U)
      request.valid.poke(true.B)
      toL2.valid.expect(true.B)
      request.ready.expect(false.B)
      dut.clock.step(2)
      toL2.bits.opcode.expect(4.U) // Get
      val tag = toL2.bits.source.peek().litValue
      toL2.ready.poke(true.B)
      request.ready.expect(true.B)
      dut.clock.step()

      // A second AMO on the same line must not slip past the pending update.
      request.bits.source.poke(2.U)
      request.bits.size.poke(1.U)
      request.ready.expect(false.B)
      toL2.valid.expect(false.B)
      fromL2.bits.opcode.poke(1.U)
      fromL2.bits.param.poke(0.U)
      fromL2.bits.source.poke(tag.U)
      fromL2.bits.data.poke(oldLine.U)
      fromL2.valid.poke(true.B)
      fromL2.ready.expect(true.B)
      dut.clock.step()
      fromL2.valid.poke(false.B)

      toL2.ready.poke(false.B)
      for (_ <- 0 until 3) {
        toL2.valid.expect(true.B)
        toL2.bits.opcode.expect(1.U) // PutPartialData
        toL2.bits.address.expect(0x100.U)
        toL2.bits.size.expect(2.U)
        toL2.bits.mask.expect(mask.U)
        toL2.bits.data.expect((BigInt(26) << (32 * word)).U)
        result.valid.expect(false.B)
        dut.clock.step()
      }
      toL2.ready.poke(true.B)
      dut.clock.step()
      toL2.valid.expect(false.B)
      fromL2.bits.opcode.poke(0.U)
      fromL2.bits.data.poke(0.U)
      fromL2.valid.poke(true.B)
      fromL2.ready.expect(true.B)
      dut.clock.step()
      fromL2.valid.poke(false.B)
      for (_ <- 0 until 3) {
        request.ready.expect(false.B)
        result.valid.expect(true.B)
        result.bits.source.expect(1.U)
        result.bits.data.expect(oldLine.U)
        dut.clock.step()
      }
      result.ready.poke(true.B)
      dut.clock.step()
      result.valid.expect(false.B)
      request.ready.expect(true.B)
    }
  }
}
